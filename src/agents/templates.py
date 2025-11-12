def get_agent_base_template(task_model: str) -> str:
    """
    Returns the f-string template for a new agent, formatted with the task_model.
    """
    
    return f"""
import os
import operator
import sys
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# --- 1. Setup ---
# Load environment variables (e.g., OPENAI_API_KEY)
# This is good practice in case the agent is run standalone.
load_dotenv()

# CRITICAL: This model is set by the evolutionary process.
# DO NOT CHANGE THIS LINE.
llm = ChatOpenAI(model="{task_model}")

# --- 2. Tool Definition ---
# This section will be replaced by the Creator Agent

@tool
def dummy_tool(input: str) -> str:
    \"\"\"A dummy tool. Use when the user asks for a 'test'.\"\"\"
    # This print is for agent debugging, not evolution logging
    # print(f"\\n--- Executing Tool: dummy_tool('{{input}}') ---") 
    return "This is a dummy response from the tool."

tools = [dummy_tool]
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)

# --- 3. Graph State Definition ---

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

# --- 4. Graph Node Definitions ---

def agent_node(state):
    \"\"\"
    The main agent node. Calls the LLM to decide the next action.
    \"\"\"
    # === SYSTEM PROMPT ===
    # This will be replaced by the Creator/Developer Agent
    system_prompt = SystemMessage(
        content="You are a helpful assistant. You have access to a 'dummy_tool'."
    )
    # === END SYSTEM PROMPT ===

    messages_for_api = [system_prompt] + state["messages"]
    
    # Call the LLM
    response = llm_with_tools.invoke(messages_for_api)
    
    return {{"messages": [response]}}

# --- 5. Conditional Edge Logic ---

def should_continue(state):
    \"\"\"
    This is the router. It checks the last message from the 'agent_node'.
    \"\"\"
    last_message = state["messages"][-1]

    # If the LLM response contains tool calls, route to 'call_tool'
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "call_tool"
    
    # Otherwise, the conversation is over
    return END

# --- 6. Graph Construction & Compilation ---

workflow = StateGraph(AgentState)

# Add the nodes
workflow.add_node("agent", agent_node)
workflow.add_node("call_tool", tool_node)

# Set the entry point
workflow.set_entry_point("agent")

# Add the conditional edges
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {{"call_tool": "call_tool", END: END}}
)

# Add the edge from tool use back to the agent
workflow.add_edge("call_tool", "agent")

# Compile the graph
app = workflow.compile()

# --- 7. Execution Function (for importability) ---

def run_agent(question: str) -> str:
    \"\"\"
    Executes the agent for a single question and returns the final answer.
    This function is required by the DGM evaluation loop.
    \"\"\"
    inputs = {{"messages": [HumanMessage(content=question)]}}
    final_response_content = "(No response from agent)"
    
    try:
        # Use app.stream with a recursion limit for safety
        for event in app.stream(inputs, {{"recursion_limit": 10}}):
            if "agent" in event:
                last_message = event["agent"]["messages"][-1]
                # Check if this is a final AIMessage (not a tool call)
                if isinstance(last_message, AIMessage) and not last_message.tool_calls:
                    final_response_content = last_message.content
                    
    except Exception as e:
        # Agent-level errors are caught and returned as a string
        # This prevents the whole evolution loop from crashing
        return f"Runtime Error: {{str(e)}}"
    
    return final_response_content

# --- 8. Main Block (for executable script) ---

if __name__ == "__main__":
    print("=" * 30)
    print(f"Agent ({task_model}) Ready.")
    print("Type 'exit' to quit.")
    print("=" * 30)

    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        
        response = run_agent(user_input)
        print(f"\\nAgent: {{response}}")
        print("-" * 30)
"""