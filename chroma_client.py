import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()


class ChromaClient:
    def __init__(self):
        # Initialize embedding model
        self._openai_ef = OpenAIEmbeddings()

        # Validate if the embedding model initialized
        if not self._openai_ef:
            raise ValueError("OpenAIEmbeddings failed to initialize")

        # Initialize vector store
        self.vector_store = Chroma(
            persist_directory="./db",
            collection_name="rag",
            embedding_function=self._openai_ef,
        )

        self._load_pdfs()

    def _load_pdfs(self):
        # Loop through all files in the resources directory
        for filename in os.listdir("resources"):
            # Before chunking check the vector store to see if the document has already been processed
            result = self.vector_store.get(where={"source": "resources/" + filename})

            if result["ids"]:
                print("Skipping " + filename + " because it has already been processed")
                continue

            # Create a document loader for the file
            loader = PyPDFLoader("resources/" + filename)

            # Create chunks of text from the document
            chunks = loader.load_and_split()

            # Add the chunks to the vector store
            self.vector_store.add_documents(documents=chunks)
