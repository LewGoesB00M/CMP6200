# Used to load all PDF files from a directory.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader 

# Used to split the PDFs into chunks for embedding.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Needed for LangChain's "Document" data type. When a PDF is loaded, it is loaded
# as a Document. 
from langchain.schema import Document

# Used to call OpenAI embedding models on the chunks.
from langchain_openai import OpenAIEmbeddings

# Used to create a FAISS vector database.
from langchain_community.vectorstores import FAISS

# Used to get the OpenAI API key from the system environment variables,
# as it is a major security breach if it is publicly accessible on this Github repo.
import os

# If the database already exists and this file is run again, the existing
# database should be cleared to avoid any potential issues. Shutil is 
# a built-in Python module with the "rmtree" method to delete a directory
# and its subdirectories.
import shutil


# Set the paths for the FAISS DB, as well as the PDFs that will be loaded.
FAISS_PATH = "FAISS-SmallChunks"
DATA_PATH = "Data/Policies"#/TESTING" # minimise token use

# Loads all PDFs, chunks them, then saves them to a FAISS DB.
# (write a dissertation section on FAISS, potentially updating lit review.)
def generate_data_store():
    # Load every PDF.
    documents = load_documents()
    # Chunk all the text.
    chunks = split_text(documents)
    # Save them to the Faiss DB.
    save_to_faiss(chunks)

# Loads every PDF from the data path.
def load_documents():
    # In the data path, load every (signified by asterisk) PDF file.
    # Because they're PDF files, the PyPDFLoader can be used to load each.
    loader = DirectoryLoader(DATA_PATH,  glob = "*.pdf",
                             loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

# Uses LangChain's RecursiveCharacterTextSplitter to split the documents into chunks
# for embedding.
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Can be changed, just a sample number. # ? Originally 1000
        chunk_overlap=100, # Helps data to not be split over multiple chunks. # ? Originally 200
        length_function=len, 
        add_start_index=True,
    )

    # Save the split chunks.
    chunks = text_splitter.split_documents(documents)
    
    # Example: "Split 100 documents into 700 chunks"
    # Just for verification that the script ran.
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")


    print(chunks[10].page_content)

    # Return the chunks so that they can be embedded and saved.
    return chunks


def save_to_faiss(chunks: list[Document]):
    # Clear out the database first if it exists.
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

    # Create a new DB from the documents.
    faiss = FAISS.from_documents(
        chunks, 
        OpenAIEmbeddings(
            model = "text-embedding-3-small", # Cost-efficient
            openai_api_key = os.environ["OPENAI_API_KEY"]
            )
    )
    
    # FAISS saves differently to Chroma, requiring the vector store to first
    # be saved to memory then committed to a folder. FAISS indexes are ~5x smaller
    # than Chroma equivalents.
    faiss.save_local(folder_path = FAISS_PATH)
    
    print(f"Saved {len(chunks)} chunks to {FAISS_PATH}.")

# When this file gets run, generate the FAISS DB, which by extension will load and chunk
# all documents.
if __name__ == "__main__":
    generate_data_store()