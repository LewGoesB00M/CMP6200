# Used to load all PDF files from a directory.
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader 

# Used to split the PDFs into chunks for embedding.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Needed for LangChain's "Document" data type. When a PDF is loaded, it is loaded
# as a Document. 
from langchain.schema import Document

# Used to call OpenAI embedding models on the chunks.
from langchain_openai import OpenAIEmbeddings

# Used to create a Chroma vector database.
from langchain_community.vectorstores import Chroma

# Used to get the OpenAI API key from the system environment variables,
# as it is a major security breach if it is publicly accessible on this Github repo.
import os

# If the database already exists and this file is run again, the existing
# database should be cleared to avoid any potential issues. Shutil is 
# a built-in Python module with the "rmtree" method to delete a directory
# and its subdirectories.
import shutil

# Set the paths for the Chroma DB, as well as the PDFs that will be loaded.
CHROMA_PATH = "Chroma"
DATA_PATH = "Data/Policies"

# Loads all PDFs, chunks them, then saves them to a Chroma DB.
# (write a dissertation section on Chroma, potentially updating lit review.)
def generate_data_store():
    # Load every PDF.
    documents = load_documents()
    # Chunk all the text.
    chunks = split_text(documents)
    # Save them to the Chroma DB.
    save_to_chroma(chunks)

# Loads every PDF from the data path.
def load_documents():
    # In the data path, load every (signified by asterisk) PDF file.
    # Because they're PDF files, the PyPDFLoader should be used to load each.
    ### UnstructuredPDFLoader can also be used, but I was unable to get it working.
    loader = DirectoryLoader(DATA_PATH,  glob="*.pdf", loader_cls = PyPDFLoader)
    documents = loader.load()
    return documents

# Uses LangChain's RecursiveCharacterTextSplitter to split the documents into chunks
# for embedding.
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, # Can be changed, just a sample number.
        chunk_overlap=100, # The overlap between chunks. Helps to prevent data being split in two.
        length_function=len, 
        add_start_index=True,
    )

    # Save the split chunks.
    chunks = text_splitter.split_documents(documents)
    
    # Example: "Split 100 documents into 700 chunks"
    # Just for verification that the script ran.
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Return the chunks so that they can be embedded and saved.
    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first if it exists.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    Chroma.from_documents(
        chunks, 
        OpenAIEmbeddings(
            model = "text-embedding-3-small", # Cost-efficient
            openai_api_key = os.environ["OPENAI_API_KEY"]
            ),
        persist_directory=CHROMA_PATH
    )
    
    # It's typically expected to assign the Chroma.from_documents() call to a variable
    # and then call persist() on it. However, this is no longer necessary as this is an automatic
    # process as of Chroma 0.4.x (according to the LangChainDeprecationWarning that gets thrown.)
    
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

# When this file gets run, generate the Chroma DB, which by extension will load and chunk
# all documents.
if __name__ == "__main__":
    generate_data_store()