import streamlit as st
import os
import traceback

from components.ui import sidebar, main_content

# import streamlit.watcher.local_sources_watcher
# original_get_module_paths = streamlit.watcher.local_sources_watcher.get_module_paths

# def patched_get_module_paths(module):
#     if module.__name__ == "torch._classes" or module.__name__.startswith("torch._classes."):
#         return []
#     return original_get_module_paths(module)

# streamlit.watcher.local_sources_watcher.get_module_paths = patched_get_module_paths

# os.environ["STREAMLIT_SERVER_WATCH_MODULES"] = "false"

from utils.silencer import silence_common_warnings
silence_common_warnings()

from utils.patch import apply_streamlit_patches
apply_streamlit_patches()

st.set_page_config(
    page_title="Business Document Assistant",
    page_icon="📄",
    layout="wide",
)

# Initializing session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
if "interface_ready" not in st.session_state:
    st.session_state.interface_ready = False
if "startup_error" not in st.session_state:
    st.session_state.startup_error = None

# Initializing models on startup
try:
    from processors.document_processor import initialize_models, check_vectorstore_exists, load_vectorstore
    from components.graph import initialize_graph

    # Function to initialize everything
    def initialize_all_components():
        """Initialize all required components for the chat interface"""
        try:
            # Pre-load models
            with st.spinner("Loading document processing models..."):
                initialize_models()
                st.session_state.models_loaded = True
            
            # Initializing graph even if no documents are present yet
            if st.session_state.graph is None:
                with st.spinner("Initializing agent graph..."):
                    st.session_state.graph = initialize_graph()
            
            # Check if documents already exist and load them
            if check_vectorstore_exists() and st.session_state.vectorstore is None:
                with st.spinner("Loading existing document database..."):
                    vectorstore = load_vectorstore()
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.retriever = vectorstore.as_retriever()
            
            st.session_state.interface_ready = True
            return True
        except Exception as e:
            st.session_state.startup_error = str(e)
            st.session_state.startup_error_traceback = traceback.format_exc()
            return False

    # Making sure all components are initialized on startup
    if not st.session_state.interface_ready:
        initialize_all_components()

    def main():
        """Main app entry point."""
        if st.session_state.startup_error:
            st.error("An error occurred during startup:")
            st.code(st.session_state.startup_error)
            with st.expander("Show detailed error traceback"):
                st.code(st.session_state.startup_error_traceback)
            
            if st.button("Retry Initialization"):
                st.session_state.startup_error = None
                st.session_state.interface_ready = False
                st.rerun()
        else:
            # Rendering the sidebar and main content
            sidebar()
            main_content()

except Exception as e:
    # Handling import errors
    st.error(f"Failed to initialize application: {str(e)}")
    st.code(traceback.format_exc())
    
    def main():
        """Fallback main function"""
        st.title("Error Loading Application")
        st.error("The application could not be initialized properly.")
        if st.button("Retry"):
            st.rerun()

if __name__ == "__main__":
    main() 