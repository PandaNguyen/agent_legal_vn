import json
import logging

def main():
    corpus_file = "./data/corpus_raw.json"
    output_file = "./data/corpus_final.json"

    print(f"Reading from {corpus_file}...")
    try:
        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {corpus_file}")
        return

    fixed_count = 0

    for doc in corpus:
        units = doc.get("content", [])
        for unit in units:
            title = unit.get("unit_title", "")
            
            # Nhận diện title bị crawl nhầm
            if "Độc lập - Tự do - Hạnh phúc" in title:
                contents = unit.get("unit_content", [])
                
                # Check nếu có content và phần tử đầu tiên là text
                if contents and isinstance(contents, list) and len(contents) > 0:
                    first_content = contents[0]
                    
                    if first_content.get("type") == "text":
                        # Lấy data của phần tử đầu tiên làm title mới
                        new_title = first_content.get("data", "")
                        
                        unit["unit_title"] = new_title
                        
                        # Xóa phần tử đầu tiên ra khỏi unit_content
                        # vì nó đã được dùng làm tiêu đề
                        unit["unit_content"] = contents[1:]
                        
                        fixed_count += 1

    print(f"Total titles fixed: {fixed_count}")

    print(f"Saving to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)
        
    print("Done ✅")

if __name__ == "__main__":
    main()
