import streamlit as st
from typing import Dict, List, Any, Optional

class GraphTracer:
    """
    Utility class for tracing and displaying graph execution steps
    """
    def __init__(self):
        """Initialize the graph tracer"""
        if "graph_trace" not in st.session_state:
            st.session_state.graph_trace = []
        if "current_node" not in st.session_state:
            st.session_state.current_node = None
        if "node_decisions" not in st.session_state:
            st.session_state.node_decisions = {}
            
    def clear_trace(self):
        """Clear the current trace"""
        st.session_state.graph_trace = []
        st.session_state.current_node = None
        st.session_state.node_decisions = {}
    
    def add_trace(self, node_name: str, state: Dict[str, Any] = None, decision: Optional[str] = None):
        """
        Add a trace entry for node execution
        
        Args:
            node_name: Name of the node
            state: The current graph state (optional)
            decision: Decision made by this node (optional)
        """
        st.session_state.current_node = node_name
        
        trace_entry = {
            "node": node_name,
            "timestamp": st.session_state.get("trace_counter", 0),
        }
        
        if state:
            trace_entry["state_info"] = {}
            if "question" in state and state["question"]:
                trace_entry["state_info"]["question"] = state["question"]
            if "documents" in state and state["documents"]:
                trace_entry["state_info"]["document_count"] = len(state["documents"])
            if "generation" in state and state["generation"]:
                gen = state["generation"]
                trace_entry["state_info"]["generation"] = (gen[:100] + "...") if len(gen) > 100 else gen
        

        if decision:
            trace_entry["decision"] = decision
            st.session_state.node_decisions[node_name] = decision
        
        st.session_state.graph_trace.append(trace_entry)

        st.session_state["trace_counter"] = st.session_state.get("trace_counter", 0) + 1
    
    def get_trace(self) -> List[Dict[str, Any]]:
        """Get the current trace"""
        return st.session_state.graph_trace
    
    def get_current_node(self) -> Optional[str]:
        """Get the currently executing node"""
        return st.session_state.current_node
    
    def get_node_decision(self, node_name: str) -> Optional[str]:
        """Get the decision made by a node"""
        return st.session_state.node_decisions.get(node_name)

graph_tracer = GraphTracer() 