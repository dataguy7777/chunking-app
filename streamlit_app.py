import os
import logging
import streamlit as st
from typing import List, Tuple
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings  # Corrected import
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import PyPDF2
import docx

# Configure Logging
logging.basicConfig(
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_documents(file_path: str) -> List[str]:
    """
    Load documents from a file (supports .txt, .pdf, .docx).

    Args:
        file_path (str): Path to the file.

    Returns:
        List[str]: List containing the document text.

    Example:
        >>> load_documents('sample.txt')
        ['This is the content of sample.txt']
    """
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                documents = [file.read()]
        elif file_path.endswith('.pdf'):
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"
                documents = [text]
        elif file_path.endswith('.docx'):
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            documents = [text]
        else:
            documents = []
            logger.warning(f"Unsupported file type: {file_path}")
        logger.info(f"Loaded document from {file_path}")
        return documents
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        return []

def split_documents(documents: List[str], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split documents into smaller chunks.

    Args:
        documents (List[str]): List of document texts.
        chunk_size (int, optional): Size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): Overlap between chunks. Defaults to 200.

    Returns:
        List[str]: List of document chunks.

    Example:
        >>> split_documents(['This is a sample document.'])
        ['This is a sample document.']
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split documents into {len(chunks)} chunks")
    return chunks

def enrich_chunks(chunks: List[str], source: str = 'unknown') -> List[Tuple[str, dict]]:
    """
    Enrich chunks with metadata.

    Args:
        chunks (List[str]): List of document chunks.
        source (str, optional): Source identifier for the chunks. Defaults to 'unknown'.

    Returns:
        List[Tuple[str, dict]]: List of tuples containing chunk and its metadata.

    Example:
        >>> enrich_chunks(['Sample chunk'])
        [('Sample chunk', {'source': 'unknown', 'length': 12})]
    """
    enriched = []
    for chunk in chunks:
        metadata = {
            'source': source,
            'length': len(chunk)
        }
        enriched.append((chunk, metadata))
    logger.info(f"Enriched {len(enriched)} chunks with metadata")
    return enriched

def embed_chunks(chunks_with_metadata: List[Tuple[str, dict]], embedding_model: str) -> Tuple[List[List[float]], List[dict]]:
    """
    Generate embeddings for each chunk using the selected embedding model.

    Args:
        chunks_with_metadata (List[Tuple[str, dict]]): List of tuples containing chunk and metadata.
        embedding_model (str): Selected embedding model identifier.

    Returns:
        Tuple[List[List[float]], List[dict]]: Tuple containing list of embeddings and corresponding metadata.

    Example:
        >>> embed_chunks([('Sample chunk', {'source': 'test'})], 'huggingface/all-MiniLM-L6-v2')
        ([[0.1, 0.2, ...]], [{'source': 'test'}])
    """
    try:
        if embedding_model.startswith("openai"):
            embeddings = OpenAIEmbeddings()
            texts = [chunk for chunk, _ in chunks_with_metadata]
            embedded = embeddings.embed_documents(texts)
        elif embedding_model.startswith("huggingface"):
            model_name = embedding_model.split(":", 1)[1]
            model = SentenceTransformer(model_name)
            texts = [chunk for chunk, _ in chunks_with_metadata]
            embedded = model.encode(texts, convert_to_numpy=True).tolist()
        else:
            logger.error(f"Unsupported embedding model selected: {embedding_model}")
            return [], []
        
        metadata = [meta for _, meta in chunks_with_metadata]
        logger.info(f"Generated embeddings using model: {embedding_model}")
        return embedded, metadata
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return [], []

def create_vector_store(embeddings: List[List[float]], metadata: List[dict], index_path: str = 'faiss.index') -> None:
    """
    Create and save a FAISS vector store.

    Args:
        embeddings (List[List[float]]): List of embeddings.
        metadata (List[dict]): List of metadata dictionaries.
        index_path (str, optional): Path to save the FAISS index. Defaults to 'faiss.index'.

    Returns:
        None

    Example:
        >>> create_vector_store([[0.1, 0.2]], [{'source': 'test'}])
        None
    """
    try:
        if not embeddings:
            logger.warning("No embeddings to add to the vector store.")
            return

        dimension = len(embeddings[0])
        index = faiss.IndexFlatL2(dimension)
        normalized_embeddings = faiss.normalize_L2(np.array(embeddings).astype('float32'))
        index.add(normalized_embeddings)
        faiss.write_index(index, index_path)
        with open('metadata.pkl', 'wb') as meta_file:
            pickle.dump(metadata, meta_file)
        logger.info(f"Vector store created with {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")

def process_file(file_path: str, embedding_model: str) -> None:
    """
    Complete processing pipeline for a single file.

    Args:
        file_path (str): Path to the file.
        embedding_model (str): Selected embedding model identifier.

    Returns:
        None

    Example:
        >>> process_file('sample.txt', 'huggingface/all-MiniLM-L6-v2')
        None
    """
    documents = load_documents(file_path)
    if not documents:
        logger.warning(f"No documents loaded from {file_path}. Skipping processing.")
        return

    chunks = split_documents(documents)
    enriched = enrich_chunks(chunks, source=file_path)
    embeddings, metadata = embed_chunks(enriched, embedding_model)
    if embeddings:
        create_vector_store(embeddings, metadata)

def process_folder(folder_path: str, embedding_model: str) -> None:
    """
    Complete processing pipeline for all files in a folder.

    Args:
        folder_path (str): Path to the folder.
        embedding_model (str): Selected embedding model identifier.

    Returns:
        None

    Example:
        >>> process_folder('./documents', 'openai:text-embedding-ada-002')
        None
    """
    try:
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                logger.info(f"Processing file: {file_path}")
                process_file(file_path, embedding_model)
    except Exception as e:
        logger.error(f"Error processing folder: {e}")

# Streamlit App
def main():
    st.title("Document Enrichment and Vectorization Tool")

    st.write("""
    ## Import your documents and enrich them with metadata before storing in a vector database.
    """)

    # Embedding Model Selection
    st.sidebar.header("Embedding Model Selection")
    embedding_options = {
        "Hugging Face - all-MiniLM-L6-v2": "huggingface:all-MiniLM-L6-v2",
        "Hugging Face - distilbert-base-nli-stsb-mean-tokens": "huggingface:distilbert-base-nli-stsb-mean-tokens",
        "OpenAI - text-embedding-ada-002": "openai:text-embedding-ada-002",
        # Add more models as needed
    }
    selected_embedding = st.sidebar.selectbox(
        "Select Embedding Model",
        options=list(embedding_options.keys()),
        index=0
    )
    embedding_model_id = embedding_options[selected_embedding]

    st.write(f"### Selected Embedding Model: **{selected_embedding}**")

    option = st.radio("Select input type:", ("Upload File", "Select Folder"))

    if option == "Upload File":
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf", "docx"])
        if uploaded_file is not None:
            # Save the uploaded file to a temporary location
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File {uploaded_file.name} uploaded successfully.")
            if st.button("Process File"):
                process_file(file_path, embedding_model_id)
                st.success("File processed and stored in vector database.")
    else:
        folder_path = st.text_input("Enter folder path:")
        if folder_path:
            if os.path.isdir(folder_path):
                if st.button("Process Folder"):
                    process_folder(folder_path, embedding_model_id)
                    st.success("Folder processed and stored in vector database.")
            else:
                st.error("Folder path does not exist.")

    st.write("---")
    st.write("### Logs")
    try:
        with open("app.log", "r") as log_file:
            logs = log_file.read()
        st.text_area("Application Logs", logs, height=300)
    except FileNotFoundError:
        st.write("No logs available.")

if __name__ == "__main__":
    main()
