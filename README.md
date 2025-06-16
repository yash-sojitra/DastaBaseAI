# Document RAG Assistant

A Streamlit application that combines document processing with RAG (Retrieval Augmented Generation) capabilities to create an intelligent chatbot that can answer questions about your documents.

## Features

- **Multiple Document Upload**: Upload multiple PDFs, JPGs, or PNGs for processing
- **Document Processing**: Conversion of documents to structured data with metadata extraction
- **RAG Chat Interface**: Ask questions about your documents and get AI-generated answers
- **Smart Retrieval**: Uses vector search to find relevant document chunks
- **Hallucination Prevention**: Checks if answers are grounded in the document content

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -U streamlit langchain_community tiktoken langchain-openai langchain-ollama langchain-chroma langchain-cohere langchainhub chromadb langchain langgraph marker-pdf
```

3. Set your Azure OpenAI API credentials as environment variables:

```bash
export AZURE_OPENAI_API_KEY=your_key
export AZURE_OPENAI_ENDPOINT=your_endpoint
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage

1. Upload your documents using the sidebar
2. Click "Process Documents" to extract information and create the vectorstore
3. Once processing is complete, use the chat interface to ask questions about your documents
4. The AI assistant will retrieve relevant information and generate answers

## Requirements

- Python 3.8+
- Azure OpenAI API access
- Ollama with llama3.2 for embeddings

## System Architecture

The system consists of two main components:

1. **Document Processing Pipeline**:
   - Document conversion to text/markdown
   - Structured data extraction via LLM
   - Text chunking and embedding

2. **RAG Chat Agent**:
   - Multi-step workflow using LangGraph
   - Document retrieval based on questions
   - Relevance assessment of retrieved documents
   - Answer generation with grounding checks 