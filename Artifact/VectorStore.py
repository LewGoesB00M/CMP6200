# Used to load all PDF files from a directory.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader 

# Used to split the PDFs into chunks for embedding.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Used to call OpenAI embedding models on the chunks.
from langchain_openai import OpenAIEmbeddings

# Used to create a FAISS vector database.
from langchain_community.vectorstores import FAISS

# Used to check if the database's path already exists,
# and also to get the OpenAI API key from system env variables.
import os

# If the database already exists and this file is run again, the existing
# database should be cleared to avoid any potential issues. Shutil is 
# a built-in Python module with the "rmtree" method to delete a directory
# and its subdirectories.
import shutil


# Set the paths for the FAISS DB, as well as the PDFs that will be loaded.
    # Options (all begin with "VectorStores/"):
        #   FAISS: Chunk size 1000, Overlap 200, PyPDFLoader with default args.
        #   FAISS-SmallChunks: Chunk size 500, Overlap 100, PyPDFLoader with default args.
        #   FAISS-Unstructured: Chunk size 1000, Overlap 200, UnstructuredPDFLoader with default args.
        #   FAISS-BigChunks: Chunk size 1500, Overlap 300, PyPDFLoader with default args.
        #   FAISS-HugeChunks: Chunk size 2000, Overlap 500, PyPDFLoader with default args.
FAISS_PATH = "VectorStores/FAISS-HugeChunks"
DATA_PATH = "Data/Policies"#/TESTING" # minimise token use

# Loads all PDFs, chunks them, then saves them to a FAISS DB.
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

# Uses LangChain's RecursiveCharacterTextSplitter to split the documents into chunks for embedding.
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # See Lines 28 - 34 for info on size and overlap.
        chunk_overlap=500,  
        length_function=len, # A custom length function can be made, but I saw no need.
        add_start_index=True, # Stores the chunk's starting character index in its metadata.
    )

    # Save the split chunks.
    chunks = text_splitter.split_documents(documents)
    
    # Example: "Split 100 documents into 700 chunks"
    # Just for verification that the script ran.
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Return the chunks so that they can be embedded and saved.
    return chunks


def save_to_faiss(chunks):
    # Clear out the database first if it exists.
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

    # Create a new DB from the documents.
    # Takes the chunks and uses OpenAI text-embedding-3-small to embed them as vectors.
    faiss = FAISS.from_documents(
        chunks, 
        OpenAIEmbeddings(
            model = "text-embedding-3-small", # Cost-efficient
            openai_api_key = os.environ["OPENAI_API_KEY"]
            )
    )
    
    # Save the generated DB to the given path.
    faiss.save_local(folder_path = FAISS_PATH)
    
    print(f"Saved {len(chunks)} chunks to {FAISS_PATH}.")

# When this file gets run, generate the FAISS DB, which by extension will load and chunk
# all documents.
if __name__ == "__main__":
    generate_data_store()