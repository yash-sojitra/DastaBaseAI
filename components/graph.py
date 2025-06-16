from langgraph.graph import StateGraph, START, END

from nodes.state import GraphState
from nodes.router import chat_router, decide_betn_respond_retrieve_toolcall
from nodes.processor import retrieve, generate, responder
from nodes.tools import tools_node
from nodes.grader import grade_documents, transform_query, decide_to_generate, grade_generation_v_documents_and_question
from utils.graph_tracer import graph_tracer

def initialize_graph():
    """Initializing the graph for the chat agent."""

    graph_tracer.clear_trace()

    workflow = StateGraph(GraphState)
    
    workflow.add_node("chat", chat_router)
    workflow.add_node("responder", responder)
    workflow.add_node("tools", tools_node)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)

    workflow.add_edge(START, "chat")
    
    workflow.add_conditional_edges(
        "chat",
        decide_betn_respond_retrieve_toolcall,
        {
            "respond": "responder",
            "tools": "tools",
            "retrieve": "retrieve",
            "end": END,
        }
    )
    
    workflow.add_edge("tools", "chat")
    workflow.add_edge("responder", "chat")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    
    workflow.add_edge("transform_query", "retrieve")
    
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": "chat",
            "not useful": "transform_query",
        },
    )
    
    graph = workflow.compile(checkpointer=None)
    
    return graph 