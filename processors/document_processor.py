import os
import tempfile
import json
import streamlit as st
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser
from langchain_openai import AzureChatOpenAI
import chromadb
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

PERSIST_DIRECTORY = os.path.join(os.getcwd(), "chroma_db")
COLLECTION_NAME = "doc-rag-chroma"

def initialize_models():
    """Initialize all the document parsing models at startup."""
    st.info("Loading document processing models...")
    # Creating models dictionary to cache models
    if "model_dict" not in st.session_state:
        st.session_state.model_dict = create_model_dict()
    
    # Creating configuration for document processing
    config = {
        'output_format': 'markdown',
        'force_ocr': True,
        'debug': False,
    }
    config_parser = ConfigParser(config)
    
    # Caching the converter and processor
    if "converter" not in st.session_state:
        st.session_state.converter = PdfConverter(
            artifact_dict=st.session_state.model_dict,
            config=config_parser.generate_config_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )
    
    st.success("Document processing models loaded successfully!")
    return st.session_state.converter

def process_document(uploaded_file):
    """Process an uploaded document using Marker."""
    # Ensuring models are initialized
    converter = st.session_state.get("converter")
    if not converter:
        converter = initialize_models()
    
    # Saving the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    # Converting document
    try:
        rendered = converter(tmp_file_path)
        text, _, images = text_from_rendered(rendered)
        
        # Parsing document with LLM
        llm = AzureChatOpenAI(deployment_name="gpt-4-2")
        prompt = f'''
        You are an expert at parsing Markdown documents into structured JSON.
        Given a Markdown representation of a document (which may include text blocks, tables, bullet points, headings, etc.), extract all the meaningful information and organize it into a clean and logical JSON structure.
        Follow these guidelines:
        - Identify sections based on headings and their contents.
        - For any tables (e.g., services list), parse them fully into arrays of objects with appropriate fields.
        - If a value is missing for a field, omit it (do not guess or fill).
        - Keep all numeric values (amounts, quantities) cleanly extracted not in strings.
        - Preserve original wording where possible.
        - Do not hallucinate data not present in the Markdown.
        - One last important thing I want you to give response with data and metadata which contains following fields.
            - issue_date
            - due_date (optional)
            - document_type: "INVOICE", "BILL", "LEGAL", "REPORT"
        - dates should have specific format like DD Month YYYY for example 16 May 2025
        - don't include new line character in response even if whole document comes in oneline
        Output only the final JSON, nothing else.
        Format the JSON cleanly with proper nesting.
            $$$
            {text}
            $$$
        '''
        
        structured_data = llm.invoke(prompt)
        json_string = structured_data.content
        clean_json_string = json_string.replace("\n", "")
        
        # Cleaning up temporary file
        os.unlink(tmp_file_path)
        
        return {
            "text": text,
            "structured_data": json.loads(clean_json_string),
            "images": images,
            "filename": uploaded_file.name
        }
    except Exception as e:
        # Cleaning up temporary file
        os.unlink(tmp_file_path)
        st.error(f"Error processing document: {e}")
        return None

def check_vectorstore_exists():
    """Check if vectorstore exists and return documents if it does."""
    if not os.path.exists(PERSIST_DIRECTORY):
        return False
    
    try:
        # Checking if the collection exists
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collections = client.list_collections()
        
        return COLLECTION_NAME in collections
    except Exception as e:
        st.error(f"Error checking vectorstore: {e}")
        return False

def get_document_count():
    """Get the count of documents in the vectorstore."""
    if not check_vectorstore_exists():
        return 0
    
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception as e:
        st.error(f"Error getting document count: {e}")
        return 0

def load_vectorstore():
    """Load existing vectorstore."""
    if not check_vectorstore_exists():
        return None
    
    try:
        # Initializing embeddings
        embd = OllamaEmbeddings(model="llama3.2:latest")
        
        # Loading existing vectorstore
        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
        vectorstore = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=embd,
            persist_directory=PERSIST_DIRECTORY
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error loading vectorstore: {e}")
        return None

def create_vectorstore(documents):
    """Create a vectorstore from the processed documents."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Creating a persistent directory for the database
    if not os.path.exists(PERSIST_DIRECTORY):
        os.makedirs(PERSIST_DIRECTORY)
    
    # Extracting text content from the processed documents
    texts = [doc["text"] for doc in documents]
    
    # Splitting into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=500, chunk_overlap=0
    )
    
    doc_splits = []
    for i, text in enumerate(texts):
        chunks = text_splitter.create_documents([text], metadatas=[{"source": documents[i]["filename"]}])
        doc_splits.extend(chunks)
    
    try:
        embd = OllamaEmbeddings(model="llama3.2:latest")

        client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=COLLECTION_NAME,
            embedding=embd,
            client=client,
            persist_directory=PERSIST_DIRECTORY,
        )
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vectorstore: {e}")
        return None 