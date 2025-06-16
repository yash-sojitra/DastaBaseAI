import warnings
import os
import logging

def silence_common_warnings():
    """
    Silence common warning messages that clutter the console but aren't critical
    """
    warnings.filterwarnings(
        "ignore", 
        message="API key must be provided when using hosted LangSmith API",
        category=UserWarning
    )
    
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    
    logging.getLogger("chromadb.segment.impl.vector.local_persistent_hnsw").setLevel(logging.ERROR)
    
    os.environ["STREAMLIT_WATCHDOG_WARNING"] = "false"
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    logging.getLogger("transformers").setLevel(logging.ERROR) 