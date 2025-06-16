from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import tools_condition
from nodes.state import GraphState
from utils.graph_tracer import graph_tracer
import streamlit as st

def chat_router(state: GraphState) -> GraphState:
    """Chat router function to process the state and return a response."""

    graph_tracer.add_trace("chat", state)
    
    llm = AzureChatOpenAI(deployment_name="gpt-4-2")
    
    base_prompt = """You are the Intelligent Document Assistant. You will be given the entire chat history."""

    # Document context if documents are available
    doc_prompt = ""
    if st.session_state.vectorstore is not None:
        doc_prompt = """You have access to a database of documents that you can search through to answer questions.
        When asked about documents, use the 'retrieve' action to search them."""
    else:
        doc_prompt = """No documents are currently loaded in the system. If the user asks about specific document content,
        politely explain that no documents have been uploaded yet and guide them to upload documents using the sidebar."""
    
    # Tools section
    tools_prompt = """
    You are equipped with following tools:

    def add(a: float, b: float)
    - adds a and b

    def multipy(a: float, b: float)
    - multiplies a and b

    def divide(a: float, b: float)
    - divides a by b
    - ensure b is not 0

    Your job is to look at the most recent user request in context and choose exactly one of three actions:

    1. retrieve
    - You need new facts from the documents.  
    - only invoke this if and only if most recent request needs retireval AND documents are available.
    - reply with single word "retrieve"

    2. tool 
    - You have enough document data, but need to run a tool.
    - reply with single word "tool".

    3. respond
    - if any further reterival and tool calling is not required and if it seems like assistant has not responded entirely then and only then reply with a single word "respond".

    4. end
    - if assistant has responded one time and no further processing is required then reply with a single word "end".

    **Important:**  
    - give answer in one word only. 
    """
    
    # Combining prompts
    system_prompt = base_prompt + "\n" + doc_prompt + "\n" + tools_prompt

    sysmsg = SystemMessage(system_prompt)
    messages = state["messages"]
    
    route_ans = llm.invoke([sysmsg] + messages)
    
    # If retrieval is requested but no vectorstore exists, switch to respond
    final_ans = route_ans.content
    if final_ans == "retrieve" and st.session_state.vectorstore is None:
        final_ans = "respond"
    
    updated_state = {
        **state,
        "chat_router": final_ans,
    }
    
    # Adding trace for routing decision
    graph_tracer.add_trace("chat", updated_state, decision=final_ans)
    
    return updated_state

def decide_betn_respond_retrieve_toolcall(state: GraphState) -> str:
    """Decision function to route between respond, retrieve, tool or end"""
    route_ans = state["chat_router"]
    decision = None
    
    if route_ans == "respond":
        decision = "respond"
    elif route_ans == 'tool':
        # For tools, finding out which specific tool
        tool_decision = tools_condition(state)
        decision = f"tool:{tool_decision}"
    elif route_ans == 'retrieve':
        decision = "retrieve"
    elif route_ans == "end":
        decision = "end"
    else:
        decision = "respond"  # Default
    
    # Adding trace for the router decision
    graph_tracer.add_trace("router_decision", state, decision=decision)
    
    return decision.split(":")[0] if ":" in decision else decision 