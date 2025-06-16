import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain import hub
from nodes.state import GraphState
from utils.graph_tracer import graph_tracer
from langchain_core.documents import Document

def retrieve(state: GraphState) -> GraphState:
    """Retrieve documents"""

    graph_tracer.add_trace("retrieve", state)
    
    recent_message = state["messages"][-1]
    question = recent_message.content
    
    # Retrieval using the vectorstore retriever
    if st.session_state.retriever:
        documents = st.session_state.retriever.invoke(question)
        updated_state = {**state, "documents": documents, "question": question}
        
        # Adding trace after retrieval with document count
        graph_tracer.add_trace("retrieve", updated_state, 
                               decision=f"Retrieved {len(documents)} documents")
        return updated_state
    else:
        # No retriever available, creating a special document to explain the situation
        placeholder_doc = Document(
            page_content=("I don't have access to any document database right now. "
                         "Please upload documents using the sidebar to use document retrieval features."),
            metadata={"source": "system_message"}
        )
        
        updated_state = {**state, "documents": [placeholder_doc], "question": question}
        graph_tracer.add_trace("retrieve", updated_state, decision="No retriever available, using placeholder")
        return updated_state

def generate(state: GraphState) -> GraphState:
    """Generate answer"""

    graph_tracer.add_trace("generate", state)
    
    llm = AzureChatOpenAI(deployment_name="gpt-4-2")
    question = state["question"]
    documents = state["documents"]
    
    # Checking if we have placeholder document
    is_placeholder = False
    if len(documents) == 1 and documents[0].metadata.get("source") == "system_message":
        is_placeholder = True
    
    if is_placeholder:
        # Simple generation without RAG prompt (no documents)
        messages = state["messages"]
        ai_message = llm.invoke(messages)
        
        updated_state = {
            **state, 
            "documents": documents, 
            "question": question, 
            "generation": ai_message.content, 
            "messages": state["messages"] + [ai_message],
        }
    else:
        # Normal RAG generation (with documents)
        prompt = hub.pull("rlm/rag-prompt")
        formatted_prompt = prompt.format(context=documents, question=question)
        
        messages = state["messages"] + [HumanMessage(formatted_prompt)]
        ai_message = llm.invoke(messages)
        
        updated_state = {
            **state, 
            "documents": documents, 
            "question": question, 
            "generation": ai_message.content, 
            "messages": messages + [ai_message],
        }
    
    # Adding trace after generation
    graph_tracer.add_trace("generate", updated_state, 
                           decision="Generated response" + (" (without documents)" if is_placeholder else ""))
    
    return updated_state

def responder(state: GraphState):
    """Respond to the user with a standard LLM response"""

    graph_tracer.add_trace("responder", state)
    
    llm = AzureChatOpenAI(deployment_name="gpt-4-2")
    response = llm.invoke(state["messages"])
    
    updated_state = {
        **state,
        "messages": state["messages"] + [response]
    }
    
    # Adding trace after responding
    graph_tracer.add_trace("responder", updated_state, 
                           decision="Direct response")
    
    return updated_state 