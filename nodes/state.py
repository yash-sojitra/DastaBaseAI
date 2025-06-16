from typing import List, Optional, TypedDict, Annotated
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        messages: List of chat messages
        chat_router: Router decision
        question: Current question
        generation: LLM generation
        documents: List of retrieved documents
    """
    messages: Annotated[list, add_messages]
    chat_router: Optional[str]
    question: Optional[str]
    generation: Optional[str]
    documents: List[str] 