from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from typing import List, Optional
import json

class MinimalAgent:
    def __init__(
        self,
        role: str,
        goal: str,
        backstory: str,
        tools: List[BaseTool],
        llm: any,
        verbose: bool = False
    ):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.llm = llm
        self.verbose = verbose

    def execute(self, task: str) -> str:
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.role}. Your goal is {self.goal}.
Backstory: {self.backstory}

Respond in character, using your expertise to complete the task.
"""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                ),
            }
            | prompt
            | self.llm
            | OpenAIFunctionsAgentOutputParser()
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose,
            handle_parsing_errors=True
        )

        return agent_executor.invoke({"input": task})["output"]

class MinimalTask:
    def __init__(
        self,
        description: str,
        agent: MinimalAgent,
        expected_output: str,
        context: Optional[List["MinimalTask"]] = None
    ):
        self.description = description
        self.agent = agent
        self.expected_output = expected_output
        self.context = context or []

class MinimalCrew:
    def __init__(
        self,
        agents: List[MinimalAgent],
        tasks: List[MinimalTask],
        verbose: bool = False
    ):
        self.agents = agents
        self.tasks = tasks
        self.verbose = verbose

    def kickoff(self) -> str:
        results = []
        for task in self.tasks:
            # Get context from previous tasks
            context = ""
            if task.context:
                context = "\n\nContext from previous tasks:\n" + "\n".join(
                    [f"- {prev_task.description}: {results[i]}" 
                     for i, prev_task in enumerate(task.context)]
                )
            
            # Execute task with context
            task_input = task.description
            if context:
                task_input += context
            
            result = task.agent.execute(task_input)
            results.append(result)
            
            if self.verbose:
                print(f"\nTask: {task.description}")
                print(f"Result: {result}\n")
        
        return "\n\n".join([
            f"# Agent: {task.agent.role}\n{result}" 
            for task, result in zip(self.tasks, results)
        ])
