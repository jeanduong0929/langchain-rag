import os
import uuid
import chromadb

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceInferenceAPIEmbeddings
from huggingface_embedding_function import HuggingFaceEmbeddingFunction

# Load environment variables
load_dotenv()


class ChromaClient:
    def __init__(self):
        # Initialize embedding model
        self._hf_ef = HuggingFaceEmbeddingFunction(
            api_key=os.environ["HF_API_KEY"], url=os.environ["HF_EMBEDDED_URL"]
        )

        # Validate if the embedding model initialized
        if not self._hf_ef:
            raise ValueError("HuggingFace embedding function failed to initialize")

        self._chroma_client = chromadb.PersistentClient(path="./db")

        if not self._chroma_client:
            raise ValueError("Chroma client failed to initialize")

        self._load_pdfs()

        self.vector_store = Chroma(
            client=self._chroma_client,
            collection_name="powerlifting",
            embedding_function=HuggingFaceInferenceAPIEmbeddings(
                api_key=os.environ["HF_API_KEY"], api_url=os.environ["HF_EMBEDDED_URL"]
            ),
        )

    def _load_pdfs(self):
        collection = self._chroma_client.get_or_create_collection(
            name="powerlifting", embedding_function=self._hf_ef
        )

        # Loop through all files in the resources directory
        for filename in os.listdir("resources"):
            # Before chunking check the vector store to see if the document has already been processed
            result = collection.get(where={"source": "resources/" + filename})

            if result["ids"]:
                print("Skipping " + filename + " because it has already been processed")
                continue

            # Create a document loader for the file
            loader = PyPDFLoader("resources/" + filename)

            text_splitter = RecursiveCharacterTextSplitter(
                separators="\n",
                chunk_size=500,
                chunk_overlap=50,
                length_function=len,
            )

            # Create chunks of text from the document
            chunks = loader.load_and_split(text_splitter=text_splitter)

            # Flatten the chunks into a list of strings
            docs = []
            metadatas = []
            for d in chunks:
                docs.append(d.page_content)
                metadatas.append(d.metadata)

            # Chunk the docs and metadatas
            doc_chunks = [docs[i : i + 10] for i in range(0, len(docs), 10)]
            meta_chunks = [metadatas[i : i + 10] for i in range(0, len(metadatas), 10)]

            for doc_chunk, meta_chunk in zip(doc_chunks, meta_chunks):
                collection.add(
                    ids=[str(uuid.uuid4()) for _ in range(len(doc_chunk))],
                    documents=doc_chunk,
                    metadatas=meta_chunk,
                )
