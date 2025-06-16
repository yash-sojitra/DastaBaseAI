import streamlit as st
from langchain_core.messages import HumanMessage
from processors.document_processor import (
    process_document, 
    create_vectorstore, 
    check_vectorstore_exists,
    get_document_count,
)

def sidebar():
    """Create sidebar for document upload and processing."""
    st.sidebar.title("📄 Document Upload")
    
    # Showing document stats if vectorstore exists
    if check_vectorstore_exists():
        doc_count = get_document_count()
        st.sidebar.success(f"Found existing document database with {doc_count} documents")
    
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents (PDF, JPG, PNG)",
        type=["pdf", "jpg", "jpeg", "png"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        process_button = st.sidebar.button("Process Documents")
        if process_button:
            with st.sidebar.status("Processing documents..."):
                processed_docs = []
                for file in uploaded_files:
                    st.sidebar.write(f"Processing {file.name}...")
                    doc = process_document(file)
                    if doc:
                        processed_docs.append(doc)
                
                if processed_docs:
                    st.session_state.processed_docs = processed_docs
                    st.sidebar.success(f"Successfully processed {len(processed_docs)} documents!")
                    
                    # Creating vectorstore from processed documents
                    with st.sidebar.status("Creating vectorstore..."):
                        vectorstore = create_vectorstore(processed_docs)
                        if vectorstore:
                            st.session_state.vectorstore = vectorstore
                            st.session_state.retriever = vectorstore.as_retriever()
                            st.sidebar.success("Vectorstore created!")
    
    # Displaying processed documents
    if st.session_state.processed_docs:
        st.sidebar.subheader("Processed Documents")
        for i, doc in enumerate(st.session_state.processed_docs):
            with st.sidebar.expander(f"{i+1}. {doc['filename']}"):
                st.write("Document Type:", doc["structured_data"]["metadata"]["document_type"])
                if "issue_date" in doc["structured_data"]["metadata"]:
                    st.write("Issue Date:", doc["structured_data"]["metadata"]["issue_date"])
                if "due_date" in doc["structured_data"]["metadata"]:
                    st.write("Due Date:", doc["structured_data"]["metadata"]["due_date"])

def clear_chat_history():
    """Clear the chat history in session state"""
    st.session_state.messages = []
    st.rerun()

def main_content():
    """Create main content area with chat interface."""
    #container for the entire chat interface
    chat_interface = st.container()
    
    # First section: Title and status bar
    with chat_interface:
        st.title("🤖 Business Document Assistant")
        
        # Status and clear chat button
        col1, col2 = st.columns([5, 1])
        with col1:
            # Show status based on whether documents are loaded
            if st.session_state.vectorstore is not None or check_vectorstore_exists():
                st.success("Ready to answer questions about your documents!")
            else:
                st.info("No documents loaded yet. You can still chat, but I won't be able to reference specific document content.")
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_chat_history()
    
        # Second section: Messages area
        messages_area = st.container()
        with messages_area:
            # Display chat messages
            for message in st.session_state.messages:
                if message.type == "human":
                    st.chat_message("user").write(message.content)
                else:
                    st.chat_message("assistant").write(message.content)
        
        # Third section: Input area (at the bottom)
        input_area = st.container()
        with input_area:
            # Chat input - always available
            user_input = st.chat_input("Ask a question...")
            if user_input:
                # Add user message to state
                st.session_state.messages.append(HumanMessage(user_input))
                
                # Process with RAG agent
                with st.status("Thinking...") as status:
                    try:
                        # Create a new config dictionary for each invocation with a unique thread_id
                        config = {"configurable": {"thread_id": f"thread_{len(st.session_state.messages)}"}}
                        
                        # Process the message
                        response = st.session_state.graph.invoke(
                            {"messages": st.session_state.messages},
                            config=config
                        )
                        st.session_state.messages = response["messages"]
                        
                        # Force rerun to show the new messages
                        st.rerun()
                    except Exception as e:
                        status.error(f"Error: {str(e)}")
                        st.error(f"An error occurred while processing your request: {str(e)}")
                        # Add a fallback response message
                        fallback_message = HumanMessage(
                            "I'm sorry, I encountered an error while processing your question. Please try again or ask a different question."
                        )
                        st.chat_message("assistant").write(fallback_message.content)
                        st.rerun() 