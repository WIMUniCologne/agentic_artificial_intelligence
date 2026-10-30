"""
Dynamic prompt agent implementation for basic conversational AI.
"""

from operator import index
from typing import Optional
from datetime import datetime
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from agentic_ai.agents.base import BaseAgent, State
from langgraph.types import Command


class DynamicPromptAgent(BaseAgent):
    """
    A simple conversational agent that uses a dynamic prompt responds to messages without tools.
    
    This is a basic type of an agent, suitable for teaching the fundamentals
    of LangGraph and agent interactions with dynamic prompts.
    """
    

    def __init__(self, llm, name, system_prompt=None, checkpointer=None):
        super().__init__(
            llm=llm,
            name=name,
            system_prompt=self.init_prompt(system_prompt), # here we initialize the dynamic prompt
            checkpointer=checkpointer)


    def _build_graph(self):
        workflow = StateGraph(state_schema=State)

        workflow.add_node("update_system_prompt", self.update_system_prompt)
        workflow.add_node("llm_node", self.llm_node)

        workflow.add_edge(START, "update_system_prompt")
        workflow.add_edge("update_system_prompt", "llm_node")
        workflow.add_edge("llm_node", END)
        
        self.compile_graph(workflow)


    def update_system_prompt(self, state: State):
        current_messages = state["messages"]
        formatted_system_prompt = self.system_prompt.format()
        for index in range(len(current_messages) -1, -1, -1):
            if isinstance(current_messages[index], SystemMessage):
                current_messages.pop(index)
        # insert the formatted system prompt at the second to last position
        current_messages.insert(-1, SystemMessage(content=formatted_system_prompt))
        return Command(update={"messages": current_messages})

    def llm_node(self, state: State):        
        current_messages = state["messages"]
        response = self.llm.invoke(current_messages)
        return {"messages": [response]}


    def run(self, message: str, thread_id: str = "default", user_id: Optional[str] = None, **kwargs) -> str:
        """
        Run the agent with a message.
        
        Args:
            message: User message to process
            thread_id: Unique identifier for the conversation thread.
            user_id: Optional unique identifier for the user (for long-term memory).
            **kwargs: Additional arguments
            
        Returns:
            Agent's response
        """
        if not self.graph:
            raise RuntimeError("Graph not built. Call _build_graph() and compile_graph() first.")
        
        config = {"configurable": {"thread_id": thread_id}}
        if user_id:
            config["configurable"]["user_id"] = user_id
        
        initial_state = {"messages": [HumanMessage(content=message)]}
        result = self.graph.invoke(initial_state, config)

        return result


    # Function to initialize the dynamic prompt
    def init_prompt(self, prompt: str) -> str:
        """
        Load a dynamic prompt and initialize it.
        """
        # Format the prompt template
        system_prompt_template = PromptTemplate(
            input_variables=["name", "date"],
            template=prompt
        )

        def get_current_time():
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def get_current_name():
            return self.name

        partial_system_prompt = system_prompt_template.partial(date=get_current_time, name=get_current_name)

        return partial_system_prompt


# Example usage and testing
if __name__ == "__main__":
    import time
    from dotenv import load_dotenv
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import InMemorySaver
    from agentic_ai.prompts.prompts import system_prompt

    load_dotenv()
    llm = init_chat_model("gemini-2.5-flash-lite", model_provider="google_genai")
    memory = InMemorySaver()
    dynamic_agent = DynamicPromptAgent(llm=llm, name="DynamicAgent", system_prompt=system_prompt, checkpointer=memory)
    dynamic_agent.run("Hello! What is your name and what time is it?", thread_id="test_thread")['messages'][-1].pretty_print()
    dynamic_agent.name = "OtherNameAgent"
    print("\nWaiting 2 seconds...")
    time.sleep(2)
    dynamic_agent.run("What is your name and what time is it?", thread_id="test_thread")['messages'][-1].pretty_print()