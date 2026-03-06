from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from paralegal_agent.config.config import settings
from typing import List, Optional
from paralegal_agent.provider.llm_factory import create_llm

@CrewBase
class EvaluateCrew:
    """Evaluate Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agent_evaluate.yaml"
    tasks_config = "config/task_evaluate.yaml"
    def __init__(
        self,
        llm_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.llm_model   = llm_model   if llm_model   is not None else settings.llm_model
        self.temperature = temperature if temperature is not None else settings.temperature
        self.max_tokens  = max_tokens  if max_tokens  is not None else settings.max_tokens
        self.llm = create_llm(
            model_name=self.llm_model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
    @agent
    def router_evaluator(self) -> Agent:
        return Agent(
            config=self.agents_config["router_evaluator"],
            llm=self.llm,
        )
    @task
    def evaluate_rag_response(self) -> Task:
        return Task(
            config=self.tasks_config["evaluate_rag_response"],  
        )
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
    
