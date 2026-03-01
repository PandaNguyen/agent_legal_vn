

from __future__ import annotations
import sys
sys.path.insert(0, ".")
import signal
import asyncio
import json
import logging
import os
import random
import re
import shutil
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urljoin, urlparse, urlunparse

try:
    from curl_cffi.requests import AsyncSession
    _HAS_CURL_CFFI = True
except ImportError:
    _HAS_CURL_CFFI = False
    import httpx  # fallback

from bs4 import BeautifulSoup, NavigableString, Tag


# ──────────────────────────── CONFIG ────────────────────────────

class Config:
    INPUT_FILE       = "./data/law_ids.txt"
    OUTPUT_FILE      = "./data/corpus.json"
    CHECKPOINT_FILE  = "./data/crawler_checkpoint.json"
    LOG_FILE         = "./data/crawler.log"

    # Concurrency
    MAX_WORKERS      = 5          # số worker đồng thời (safe: 3–10)

    # Timing (per worker)
    BASE_DELAY       = 0.8        # giây chờ tối thiểu giữa các request
    MAX_DELAY        = 60.0       # giây chờ tối đa khi bị rate-limit
    JITTER           = 0.5        # thêm random(0, JITTER) vào delay

    # Retry
    RETRY_ATTEMPTS   = 4
    RETRY_BASE_DELAY = 2.0

    # Request
    REQUEST_TIMEOUT  = 35

    # Output
    INCLUDE_RAW_HTML = False
    SEGMENT_MODE     = "dieu_fallback_khoan"
    FETCH_AJAX_DIAGRAM = True

    # [FIX #4] Save mỗi N IDs thực sự hoàn thành (thay vì theo idx task)
    SAVE_EVERY       = 10

    SEARCH_URL = (
        "https://thuvienphapluat.vn/page/tim-van-ban.aspx?"
        "keyword={keyword}&area=2&type=0&status=0&lan=1&org=0&signer=0"
        "&match=False&sort=1&bdate=25/02/1946&edate={edate}"
    )

    UA_POOL: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    ]

    BASE_HEADERS = {
        "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language":           "vi-VN,vi;q=0.9,en;q=0.7",
        "Accept-Encoding":           "gzip, deflate, br",
        "Connection":                "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control":             "max-age=0",
    }


# ──────────────────────────── LOGGING ────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────── HELPERS ────────────────────────────

def clean(text: Optional[str]) -> str:
    return " ".join((text or "").split())


def fold_vietnamese(text: str) -> str:
    result = unicodedata.normalize("NFD", text)
    result = "".join(c for c in result if unicodedata.category(c) != "Mn")
    return result.replace("đ", "d").replace("Đ", "D")


def normalize_id(value: Optional[str]) -> str:
    return " ".join((value or "").split()).upper()


def normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        if not p.scheme or not p.netloc:
            return ""
        return urlunparse(p._replace(params="", query="", fragment="")).rstrip("/")
    except Exception:
        return ""


def decode_escaped_html(html: str) -> str:
    if not html or '\\"' not in html and "\\r\\n" not in html:
        return html
    for src, dst in [('\\"', '"'), ("\\r\\n", "\n"), ("\\n", "\n"),
                     ("\\t", "\t"), ("\\/", "/")]:
        html = html.replace(src, dst)
    return html.rstrip("\\")


def pick_ua() -> str:
    return random.choice(Config.UA_POOL)


def jitter_delay(base: float) -> float:
    return base + random.uniform(0, Config.JITTER)


# ──────────────────────────── CHECKPOINT ─────────────────────────

class CheckpointManager:
    """Lưu/đọc trạng thái đã crawl. Hỗ trợ resume. Thread-safe cho asyncio."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = asyncio.Lock()
        self.data = self._load()

    def _load(self) -> Dict:
        default = {
            "processed_ids": [],
            "failed_ids":    [],
            "last_processed": None,
            "start_time":    datetime.now().isoformat(),
            "last_update":   None,
        }
        if not self.path.exists():
            return default
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("checkpoint không phải dict")
            processed = list(dict.fromkeys(data.get("processed_ids", [])))
            processed_set = set(processed)
            failed = [x for x in dict.fromkeys(data.get("failed_ids", []))
                      if x not in processed_set]
            return {**default, **data, "processed_ids": processed, "failed_ids": failed}
        except Exception as e:
            logger.warning(f"Không thể load checkpoint: {e}")
            return default

    def _write_atomic(self):
        self.data["last_update"] = datetime.now().isoformat()
        tmp = self.path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(self.data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.path)
        except Exception as e:
            logger.error(f"Không thể lưu checkpoint: {e}")

    async def mark_processed(self, law_id: str):
        async with self._lock:
            if law_id not in self.data["processed_ids"]:
                self.data["processed_ids"].append(law_id)
            self.data["failed_ids"] = [x for x in self.data["failed_ids"] if x != law_id]
            self.data["last_processed"] = law_id
            self._write_atomic()

    async def mark_failed(self, law_id: str):
        async with self._lock:
            if law_id not in self.data["failed_ids"]:
                self.data["failed_ids"].append(law_id)
                self._write_atomic()

    def is_processed(self, law_id: str) -> bool:
        return law_id in self.data["processed_ids"]

    @property
    def processed_set(self) -> set:
        return set(self.data["processed_ids"])

    def stats(self) -> Dict:
        return {
            "processed": len(self.data["processed_ids"]),
            "failed":    len(self.data["failed_ids"]),
            "last":      self.data["last_processed"],
        }


# ──────────────────────────── ASYNC HTTP CLIENT ──────────────────
#
# [FIX #2] Mỗi worker tạo HttpClient riêng → không còn shared mutable
# state (_delay) giữa các worker chạy song song.

class HttpClient:
    """
    Async HTTP client với:
      - curl_cffi (Chrome TLS fingerprint) nếu có
      - Fallback httpx nếu curl_cffi chưa cài
      - Adaptive delay + 429-aware backoff (per-instance, không share)
      - Rotating User-Agent
    """

    def __init__(self):
        # [FIX #2] _delay là per-instance → mỗi worker có delay riêng
        self._delay = Config.BASE_DELAY
        self._session: Optional[object] = None

    async def __aenter__(self):
        headers = {**Config.BASE_HEADERS, "User-Agent": pick_ua()}
        if _HAS_CURL_CFFI:
            self._session = AsyncSession(
                impersonate="chrome124",
                headers=headers,
                timeout=Config.REQUEST_TIMEOUT,
                verify=False,
            )
            await self._session.__aenter__()
        else:
            logger.warning("curl_cffi không có sẵn, dùng httpx (TLS fingerprint không được spoof)")
            self._session = httpx.AsyncClient(
                headers=headers,
                timeout=Config.REQUEST_TIMEOUT,
                follow_redirects=True,
            )
            await self._session.__aenter__()
        return self

    async def __aexit__(self, *args):
        if self._session:
            await self._session.__aexit__(*args)

    def _rotate_ua(self):
        if hasattr(self._session, "headers"):
            self._session.headers["User-Agent"] = pick_ua()

    async def get(self, url: str, attempt: int = 0) -> Optional[str]:
        """GET url, trả về text hoặc None nếu hết retry."""
        try:
            d = jitter_delay(self._delay)
            await asyncio.sleep(d)
            self._rotate_ua()

            resp = await self._session.get(url)

            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", random.randint(10, 60)))
                wait = min(retry_after, Config.MAX_DELAY)
                logger.warning(f"429 Too Many Requests – chờ {wait:.1f}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
                self._delay = min(self._delay * 1.5, Config.MAX_DELAY / 2)
                if attempt < Config.RETRY_ATTEMPTS - 1:
                    return await self.get(url, attempt + 1)
                return None

            if resp.status_code == 403:
                logger.warning(f"403 Forbidden ({url}) – có thể bị block tạm thời")
                if attempt < Config.RETRY_ATTEMPTS - 1:
                    wait = Config.RETRY_BASE_DELAY * (2 ** attempt)
                    await asyncio.sleep(wait)
                    return await self.get(url, attempt + 1)
                return None

            resp.raise_for_status()
            self._delay = max(Config.BASE_DELAY, self._delay * 0.9)
            return resp.text

        except Exception as e:
            if attempt < Config.RETRY_ATTEMPTS - 1:
                wait = jitter_delay(Config.RETRY_BASE_DELAY * (2 ** attempt))
                logger.warning(f"Request lỗi ({url}) lần {attempt + 1}: {e} – thử lại sau {wait:.1f}s")
                await asyncio.sleep(wait)
                return await self.get(url, attempt + 1)
            logger.error(f"Hết retry: {url} – {e}")
            return None


# ──────────────────────────── PARSER ─────────────────────────────
# [UNCHANGED] Logic parse HTML giữ nguyên 100% từ v8/v9

class DocumentParser:
    """
    Parse HTML từ thuvienphapluat.vn.
    Logic hoàn toàn giữ nguyên từ LegalDocumentCrawler v8.
    """

    METADATA_LABELS = {
        "Số hiệu":       "doc_number",
        "Loại văn bản":  "doc_type",
        "Nơi ban hành":  "issuing_authority",
        "Người ký":      "signer",
        "Ngày ban hành": "issue_date",
        "Ngày hiệu lực": "effective_date",
        "Tình trạng":    "status",
        "Ngày công báo": "gazette_date",
        "Số công báo":   "gazette_number",
    }

    DIAGRAM_LABELS = {
        "Lĩnh vực":         "field",
        "Lĩnh vực quản lý": "field",
        "Lĩnh vực, ngành":  "field",
        "Ngày công báo":    "gazette_date",
        "Số công báo":      "gazette_number",
        "Cơ quan ban hành": "issuing_authority",
        "Nơi ban hành":     "issuing_authority",
    }

    CHAPTER_SECTION_PATTERNS = (
        re.compile(r"^chuong_\d+$", re.I),
        re.compile(r"^chuong_\d+_name$", re.I),
        re.compile(r"^muc_\d+(?:_\d+)?$", re.I),
        re.compile(r"^muc_\d+(?:_\d+)?_name$", re.I),
    )

    def __init__(self):
        self._meta_map    = {fold_vietnamese(k).lower(): v for k, v in self.METADATA_LABELS.items()}
        self._diagram_map = {fold_vietnamese(k).lower(): v for k, v in self.DIAGRAM_LABELS.items()}

    def _meta_key(self, label: str) -> Optional[str]:
        return self._meta_map.get(fold_vietnamese(clean(label).rstrip(":")).lower())

    def _diagram_key(self, label: str) -> Optional[str]:
        return self._diagram_map.get(fold_vietnamese(clean(label).strip(":")).lower())

    @staticmethod
    def _label_value_pairs(tokens: List[str]) -> List[Tuple[str, str]]:
        pairs, i = [], 0
        tokens = [clean(t) for t in tokens if clean(t)]
        while i < len(tokens):
            if tokens[i].endswith(":"):
                label = tokens[i][:-1].strip()
                value = tokens[i + 1].strip() if i + 1 < len(tokens) else ""
                if label:
                    pairs.append((label, value))
                i += 2
            else:
                i += 1
        return pairs

    def extract_title(self, soup: BeautifulSoup) -> str:
        for sel in ["#divThuocTinh h1", "h1.tit", "h1"]:
            tag = soup.select_one(sel)
            if tag and (t := clean(tag.get_text(" ", strip=True))):
                return t
        for meta_attr, attr_name in [("property", "og:title"), ("name", "title")]:
            tag = soup.find("meta", {meta_attr: attr_name})
            if tag and tag.get("content"):
                return clean(tag["content"])
        tag = soup.find("title")
        return clean(tag.get_text(" ", strip=True)) if tag else ""

    def extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        metadata: Dict[str, str] = {}
        table = soup.select_one("#divThuocTinh table")
        if not table:
            return metadata
        for row in table.select("tr"):
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
            for label, value in self._label_value_pairs(cells):
                key = self._meta_key(label)
                if key and value:
                    metadata[key] = clean(value)
        return metadata

    def extract_summary(self, soup: BeautifulSoup, metadata: Dict, meta_tags: Dict) -> str:
        for key in ("description", "og:description"):
            if meta_tags.get(key):
                return clean(meta_tags[key])
        tab = soup.select_one("#tab-1")
        if not tab:
            return ""
        clone = BeautifulSoup(str(tab), "html.parser")
        for bad in clone.select("script, style, #divThuocTinh"):
            bad.decompose()
        meta_markers = {"Số hiệu:", "Loại văn bản:", "Ngày ban hành:"}
        for tag in clone.select("p, div"):
            text = clean(tag.get_text(" ", strip=True))
            if len(text) >= 40 and not any(m in text for m in meta_markers):
                return text[:1200]
        fallback = clean(clone.get_text(" ", strip=True))
        for v in metadata.values():
            if v:
                fallback = fallback.replace(v, " ")
        return clean(fallback)[:1200]

    def extract_field_from_url(self, url: str) -> str:
        try:
            parts = [unquote(p) for p in urlparse(url).path.split("/") if p]
            if len(parts) >= 2 and parts[0].lower() == "van-ban":
                return clean(parts[1].replace("-", " ")).title()
        except Exception:
            pass
        return ""

    def parse_diagram_fields(self, soup: BeautifulSoup) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        viewing_doc = soup.select_one("#viewingDocument")
        if viewing_doc:
            for att in viewing_doc.select(".att"):
                hd = att.select_one(".hd")
                ds = att.select_one(".ds")
                if not hd or not ds:
                    continue
                label = clean(hd.get_text(" ", strip=True)).rstrip(":")
                value = clean(ds.get_text(" ", strip=True))
                key = self._diagram_key(label)
                if key and value and key not in fields:
                    fields[key] = value
            if not fields:
                raw_text = viewing_doc.get_text("\n", strip=True)
                for label in self.DIAGRAM_LABELS:
                    pattern = rf"{re.escape(label)}\s*:\s*([^\n]+)"
                    m = re.search(pattern, raw_text, re.I)
                    if not m:
                        continue
                    key = self._diagram_key(label)
                    value = clean(m.group(1))
                    if key and value and key not in fields:
                        fields[key] = value
        for row in soup.select("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            for label, value in self._label_value_pairs(cells):
                key = self._diagram_key(label)
                if key and value and key not in fields:
                    fields[key] = clean(value)
        for node in soup.select("p, li, div, span"):
            text = clean(node.get_text(" ", strip=True))
            if ":" not in text or len(text) > 300:
                continue
            label, value = text.split(":", 1)
            key = self._diagram_key(label)
            if key and value and key not in fields:
                fields[key] = clean(value)
        return fields

    def _find_content_root(self, soup: BeautifulSoup) -> Optional[Tag]:
        for sel in [
            "#tab1 .cldivContentDocVn .content1",
            "#divContentDoc.cldivContentDocVn .content1",
            "div.cldivContentDocVn div.content1",
        ]:
            root = soup.select_one(sel)
            if root:
                return root
        for root in soup.select("div.content1"):
            if "cldivContentDocEn" not in (root.parent.get("class") or []):
                return root
        return None

    def _get_unit_marker(self, node: Tag, prefix: str) -> Optional[str]:
        candidates = []
        if node.name == "a" and node.get("name"):
            candidates.append(node["name"].strip())
        candidates += [a["name"].strip() for a in node.select("a[name]")]
        for name in candidates:
            if name.startswith(prefix) and not name.endswith("_name"):
                return name
        return None

    def _get_unit_title(self, node: Tag, marker: str) -> str:
        if node.name == "a":
            parent = node.find_parent(["p", "li", "div"])
            if parent and (t := clean(parent.get_text(" ", strip=True))):
                return t
        if t := clean(node.get_text(" ", strip=True)):
            return t
        anchor = node.find("a", attrs={"name": marker})
        return clean(anchor.get_text(" ", strip=True)) if anchor else marker

    def _node_text(self, node: Tag) -> str:
        clone = BeautifulSoup(str(node), "html.parser").find(node.name)
        if not clone:
            return ""
        for bad in clone.select("script, style"):
            bad.decompose()
        for a in clone.select("a[name]"):
            if not a.get("href"):
                a.unwrap()
        return clean(clone.get_text(" ", strip=True))

    def _is_signature_table(self, node: Tag) -> bool:
        if node.name != "table":
            return False
        folded = fold_vietnamese(clean(node.get_text(" ", strip=True))).lower()
        return any(m in folded for m in ["noi nhan", "tm.", "kt.", "thu tuong",
                                          "pho thu tuong", "chu tich"])

    def _next_tag_sibling(self, node: Tag) -> Optional[Tag]:
        sib = node.next_sibling
        while sib is not None:
            if isinstance(sib, NavigableString):
                sib = sib.next_sibling
                continue
            if isinstance(sib, Tag):
                if sib.name in {"script", "style"}:
                    sib = sib.next_sibling
                    continue
                if sib.name == "p" and not clean(sib.get_text(" ", strip=True)):
                    sib = sib.next_sibling
                    continue
                return sib
            sib = sib.next_sibling
        return None

    def _is_footer(self, node: Tag) -> bool:
        if self._is_signature_table(node):
            return True
        if node.name == "p" and not clean(node.get_text(" ", strip=True)):
            nxt = self._next_tag_sibling(node)
            if nxt and (self._is_signature_table(nxt) or
                        (nxt.find("table") and self._is_signature_table(nxt.find("table")))):
                return True
        return False

    def _is_chapter_heading(self, node: Tag) -> bool:
        if node.name not in {"p", "li", "a"}:
            return False
        names = []
        if node.name == "a" and node.get("name"):
            names.append(node["name"].strip())
        names += [a["name"].strip() for a in node.select("a[name]")]
        return any(pat.match(n) for n in names for pat in self.CHAPTER_SECTION_PATTERNS)

    def extract_units(self, soup: BeautifulSoup) -> List[Dict]:
        root = self._find_content_root(soup)
        if not root:
            return []
        clone = BeautifulSoup(str(root), "html.parser")
        clone_root = clone.find("div", class_="content1") or clone
        for bad in clone_root.select("script, style"):
            bad.decompose()
        prefix = "dieu_"
        if Config.SEGMENT_MODE == "dieu_fallback_khoan" and not clone_root.select("a[name^='dieu_']"):
            prefix = "khoan_"
        units: List[Dict] = []
        current: Optional[Dict] = None
        seen_texts: set = set()
        for node in clone_root.find_all(["a", "p", "li", "table"], recursive=True):
            if node.name in {"p", "li"} and node.find_parent("table"):
                continue
            marker = self._get_unit_marker(node, prefix)
            if marker and (not current or marker != current["unit_id"]):
                if current and current["unit_content"]:
                    units.append(current)
                current = {
                    "unit_id":      marker,
                    "unit_title":   self._get_unit_title(node, marker),
                    "unit_content": [],
                }
                seen_texts = set()
                line = self._node_text(node)
                if line and line.lower() != current["unit_title"].lower():
                    current["unit_content"].append({"type": "text", "data": line})
                    seen_texts.add(line.lower())
                continue
            if not current:
                continue
            if self._is_footer(node):
                if current["unit_content"]:
                    units.append(current)
                current = None
                break
            if node.name == "table":
                current["unit_content"].append({"type": "table_html", "data": str(node)})
                continue
            if self._is_chapter_heading(node):
                continue
            if node.name in {"p", "li"}:
                text = self._node_text(node)
                if not text or text.lower() == current["unit_title"].lower():
                    continue
                if text.lower() in seen_texts:
                    continue
                current["unit_content"].append({"type": "text", "data": text})
                seen_texts.add(text.lower())
        if current and current["unit_content"]:
            units.append(current)
        return units

    def extract_document(self, html: str, law_id: str, doc_url: str) -> Optional[Dict]:
        try:
            soup = BeautifulSoup(decode_escaped_html(html), "html.parser")
            meta_tags: Dict[str, str] = {}
            for attr, key in [("name", "description"), ("property", "og:description")]:
                tag = soup.find("meta", {attr: key})
                if tag and tag.get("content"):
                    meta_tags[key] = clean(tag["content"])
            canonical = soup.find("link", rel="canonical")
            source_url = canonical["href"] if canonical and canonical.get("href") else doc_url
            metadata = self.extract_metadata(soup)
            title    = self.extract_title(soup)
            summary  = self.extract_summary(soup, metadata, meta_tags)
            content  = self.extract_units(soup)
            result = {
                "source_url":   source_url,
                "title":        title,
                "metadata":     metadata,
                "summary":      summary,
                "diagram_info": {},
                "content":      content,
            }
            if Config.INCLUDE_RAW_HTML:
                result["raw_html"] = html
            return result
        except Exception as e:
            logger.error(f"Lỗi extract {law_id}: {e}")
            return None


# ──────────────────────────── CRAWLER ────────────────────────────

class AsyncLegalCrawler:
    """
    Async crawler với các cải tiến:
      - asyncio.Queue thay vì gather(N tasks) [FIX #1]
      - Mỗi worker dùng HttpClient riêng [FIX #2]
      - Lock nhất quán cho results [FIX #3]
      - Counter thực tế cho save trigger [FIX #4]
      - Backup output trước khi ghi [FIX #5]
      - Checkpoint chỉ mark khi append thành công [FIX #6]
    """

    def __init__(self):
        self.checkpoint  = CheckpointManager(Config.CHECKPOINT_FILE)
        self.parser      = DocumentParser()
        # [FIX #3] Lock bao phủ cả append lẫn save
        self._results_lock  = asyncio.Lock()
        # [FIX #4] Đếm số IDs thực sự hoàn thành (dùng asyncio-safe counter)
        self._done_count    = 0
        self._done_lock     = asyncio.Lock()
        self.results: List[Dict] = self._load_existing_results()
        self.saved_doc_numbers = {
            normalize_id(r.get("metadata", {}).get("doc_number"))
            for r in self.results if isinstance(r.get("metadata"), dict)
        } - {""}
        self.saved_source_urls = {
            normalize_url(r.get("source_url", ""))
            for r in self.results
        } - {""}

    def _load_existing_results(self) -> List[Dict]:
        if not os.path.exists(Config.OUTPUT_FILE):
            return []
        try:
            with open(Config.OUTPUT_FILE, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Output file không hợp lệ, bỏ qua.")
                return []
            deduped, seen_nums, seen_urls = [], set(), set()
            for item in data:
                if not isinstance(item, dict):
                    continue
                num = normalize_id(item.get("metadata", {}).get("doc_number"))
                url = normalize_url(item.get("source_url", ""))
                if (num and num in seen_nums) or (url and url in seen_urls):
                    continue
                deduped.append(item)
                if num: seen_nums.add(num)
                if url: seen_urls.add(url)
            logger.info(f"Loaded {len(deduped)} documents từ {Config.OUTPUT_FILE}")
            return deduped
        except Exception as e:
            logger.warning(f"Không load được output cũ: {e}")
            return []

    def _find_doc_url(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "html.parser")
        for selector in [
            "#block-info-advan .content-0 .nqTitle a[href*='/van-ban/']",
            "#block-info-advan .nqTitle a[href*='/van-ban/']",
            ".left-col .nqTitle a[href*='/van-ban/']",
            "a[href*='/van-ban/']",
        ]:
            for a in soup.select(selector):
                href = (a.get("href") or "").strip()
                if "/van-ban/" not in href or "?tab=" in href:
                    continue
                if href.startswith("//"):    return "https:" + href
                if href.startswith("http"): return href
                return "https://thuvienphapluat.vn" + (href if href.startswith("/") else "/" + href)
        logger.warning("Không tìm thấy link văn bản")
        return None

    async def _fetch_diagram_info(
        self,
        client: HttpClient,
        soup: BeautifulSoup,
        doc_url: str,
        metadata: Dict,
        law_id: str,
    ) -> Dict[str, str]:
        fallback_info = {
            "field":             self.parser.extract_field_from_url(doc_url),
            "gazette_date":      metadata.get("gazette_date", ""),
            "gazette_number":    metadata.get("gazette_number", ""),
            "issuing_authority": metadata.get("issuing_authority", ""),
        }
        info = self.parser.parse_diagram_fields(soup)

        if not Config.FETCH_AJAX_DIAGRAM:
            for k, v in fallback_info.items():
                if not clean(info.get(k, "")) and clean(v):
                    info[k] = clean(v)
            return info

        scripts = " ".join(s.get_text() for s in soup.find_all("script"))
        match = re.search(r'["\'](?P<url>/AjaxLoadData/LoadLuocDo\.aspx\?[^"\']+)["\']',
                          scripts, re.I)
        if not match:
            logger.warning(f"tab4=missing_ajax law_id={law_id}")
            for k, v in fallback_info.items():
                if not clean(info.get(k, "")) and clean(v):
                    info[k] = clean(v)
            return info

        ajax_url = urljoin("https://thuvienphapluat.vn", match.group("url").replace("\\/", "/"))
        text = await client.get(ajax_url)
        if not text:
            logger.warning(f"tab4=request_failed law_id={law_id}")
        else:
            try:
                ajax_soup = BeautifulSoup(decode_escaped_html(text), "html.parser")
                extracted = self.parser.parse_diagram_fields(ajax_soup)
                if extracted:
                    info.update(extracted)
                    logger.info(f"tab4=success law_id={law_id}")
                else:
                    logger.warning(f"tab4=denied_or_empty law_id={law_id}")
            except Exception as e:
                logger.warning(f"tab4=parse_failed law_id={law_id} error={e}")

        for k, v in fallback_info.items():
            if not clean(info.get(k, "")) and clean(v):
                info[k] = clean(v)
        return info

    async def crawl_one(self, law_id: str, client: HttpClient) -> bool:
        """
        Crawl một law_id. Logic giữ nguyên từ v9, chỉ cải thiện phần
        checkpoint/dedup để tránh false-positive skip.
        """
        if self.checkpoint.is_processed(law_id):
            if normalize_id(law_id) in self.saved_doc_numbers:
                logger.info(f"Skip {law_id} – đã có")
                return True
            logger.warning(f"{law_id} trong checkpoint nhưng thiếu output, crawl lại")

        logger.info(f"Crawling: {law_id}")
        search_url = Config.SEARCH_URL.format(
            keyword=quote(law_id, safe=""),
            edate=datetime.now().strftime("%d/%m/%Y"),
        )

        search_html = await client.get(search_url)
        if not search_html:
            await self.checkpoint.mark_failed(law_id)
            return False

        doc_url = self._find_doc_url(search_html)
        if not doc_url:
            logger.error(f"Không tìm thấy URL cho: {law_id}")
            await self.checkpoint.mark_failed(law_id)
            return False

        doc_html = await client.get(doc_url)
        if not doc_html:
            await self.checkpoint.mark_failed(law_id)
            return False

        doc = self.parser.extract_document(doc_html, law_id, doc_url)
        if not doc:
            await self.checkpoint.mark_failed(law_id)
            return False

        soup = BeautifulSoup(decode_escaped_html(doc_html), "html.parser")
        doc["diagram_info"] = await self._fetch_diagram_info(
            client, soup, doc.get("source_url", doc_url), doc.get("metadata", {}), law_id
        )

        # [FIX #3] Lock bảo vệ cả append lẫn check dedup
        appended = False
        async with self._results_lock:
            num = normalize_id(doc.get("metadata", {}).get("doc_number") or law_id)
            url = normalize_url(doc.get("source_url", "") or doc_url)
            if num not in self.saved_doc_numbers and (not url or url not in self.saved_source_urls):
                self.results.append(doc)
                self.saved_doc_numbers.add(num)
                if url:
                    self.saved_source_urls.add(url)
                appended = True
            else:
                logger.info(f"Bỏ qua append {law_id} – đã tồn tại trong output")

        # [FIX #6] Chỉ mark processed khi append thành công hoặc đã tồn tại hợp lệ
        await self.checkpoint.mark_processed(law_id)
        logger.info(f"✓ Done: {law_id} (appended={appended})")
        return True

    # ── Worker loop ───────────────────────────────────────────────

    async def _worker_loop(self, queue: asyncio.Queue, worker_id: int, total: int):
        """
        [FIX #1] Mỗi worker chạy vòng lặp lấy task từ Queue.
        [FIX #2] Mỗi worker tạo HttpClient độc lập → không share _delay.
        """
        async with HttpClient() as client:
            while True:
                try:
                    law_id = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                remaining = queue.qsize()
                processed_so_far = total - remaining - 1  # xấp xỉ
                logger.info(f"[worker-{worker_id}] {law_id} (~{processed_so_far}/{total})")

                try:
                    await self.crawl_one(law_id.strip(), client)
                except Exception as e:
                    logger.error(f"[worker-{worker_id}] Lỗi không mong đợi với {law_id}: {e}")
                    await self.checkpoint.mark_failed(law_id)
                finally:
                    queue.task_done()

                # [FIX #4] Đếm theo IDs thực sự hoàn thành, không phải idx task
                async with self._done_lock:
                    self._done_count += 1
                    should_save = (self._done_count % Config.SAVE_EVERY == 0)

                if should_save:
                    async with self._results_lock:
                        self._save_results_sync()
                    s = self.checkpoint.stats()
                    logger.info(
                        f"Progress: done={self._done_count}/{total} "
                        f"processed={s['processed']} failed={s['failed']}"
                    )

    # ── Main entry ────────────────────────────────────────────────

    async def crawl_all(self, law_ids: List[str]):
        total = len(law_ids)
        logger.info(f"Bắt đầu crawl {total} văn bản | workers={Config.MAX_WORKERS}")

        # [FIX #1] Dùng Queue – chỉ tạo MAX_WORKERS coroutine, không tạo N coroutine
        queue: asyncio.Queue[str] = asyncio.Queue()
        for lid in law_ids:
            await queue.put(lid)

        workers = [
            asyncio.create_task(self._worker_loop(queue, i, total))
            for i in range(Config.MAX_WORKERS)
        ]

        await asyncio.gather(*workers, return_exceptions=True)
        await queue.join()  # đảm bảo queue thực sự trống

        # Lưu lần cuối
        async with self._results_lock:
            self._save_results_sync()

        self._print_summary()

    # ── Save helpers ──────────────────────────────────────────────

    def _save_results_sync(self):
        """
        [FIX #5] Backup file cũ trước khi ghi đè.
        Ghi atomic: temp → rename.
        Gọi trong asyncio context, phải được bảo vệ bởi _results_lock ở ngoài.
        """
        if not self.results and os.path.exists(Config.OUTPUT_FILE):
            logger.info("Không có dữ liệu mới, giữ nguyên output cũ")
            return

        output_path = Path(Config.OUTPUT_FILE)
        tmp_path    = output_path.with_suffix(".tmp")
        backup_path = output_path.with_suffix(".bak")

        try:
            # Backup file hiện tại trước khi ghi đè
            if output_path.exists():
                shutil.copy2(output_path, backup_path)

            tmp_path.write_text(
                json.dumps(self.results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp_path.replace(output_path)
            logger.info(f"Saved {len(self.results)} documents → {Config.OUTPUT_FILE}")

            # Xóa backup sau khi ghi thành công
            if backup_path.exists():
                backup_path.unlink()

        except Exception as e:
            logger.error(f"Lỗi lưu output: {e}")
            # Nếu lỗi, backup vẫn còn → có thể khôi phục thủ công

    def _print_summary(self):
        s     = self.checkpoint.stats()
        total = s["processed"] + s["failed"]
        rate  = s["processed"] / total * 100 if total else 0
        msg = (
            f"\n{'=' * 50}\n"
            f"  Processed : {s['processed']}\n"
            f"  Failed    : {s['failed']}\n"
            f"  Success % : {rate:.1f}%\n"
            f"  Output    : {Config.OUTPUT_FILE}\n"
            f"{'=' * 50}"
        )
        logger.info(msg)
        print(msg)


# ──────────────────────────── MAIN ───────────────────────────────

async def main():
    if not os.path.exists(Config.INPUT_FILE):
        print(f"⚠  Tạo file '{Config.INPUT_FILE}' với danh sách ID luật (mỗi ID một dòng)")
        print("Ví dụ:\n  18/2023/NĐ-CP\n  97/2023/NĐ-CP")
        return

    with open(Config.INPUT_FILE, encoding="utf-8") as f:
        law_ids = [line.strip() for line in f if line.strip()]

    if not law_ids:
        logger.error("File input rỗng!")
        return

    if not _HAS_CURL_CFFI:
        logger.warning(
            "⚠  curl_cffi chưa được cài. Để đạt hiệu quả tốt nhất:\n"
            "   pip install 'curl_cffi>=0.7.0'\n"
            "   Đang dùng httpx làm fallback..."
        )

    logger.info(f"Loaded {len(law_ids)} IDs từ {Config.INPUT_FILE}")
    crawler = AsyncLegalCrawler()
    loop = asyncio.get_event_loop()
    def _shutdown(sig, frame):
        logger.warning(f"Nhận signal {sig}, đang lưu và thoát...")
        # Lưu đồng bộ ngay lập tức
        crawler._save_results_sync()
        logger.warning("Đã lưu. Có thể thoát an toàn.")
        sys.exit(0)
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)
    await crawler.crawl_all(law_ids)


if __name__ == "__main__":
    asyncio.run(main())