import requests
from chromadb import Documents, Embeddings, EmbeddingFunction
from dotenv import load_dotenv


class HuggingFaceEmbeddingFunction(EmbeddingFunction):
    # Default constructor
    def __init__(self, url="", api_key=""):
        load_dotenv()
        self._url = url
        self._api_key = api_key

        if not self._url or not self._api_key:
            raise ValueError("URL and API_KEY must be specified")

        self._session = requests.Session()

    def __call__(self, input: Documents) -> Embeddings:
        response = self._session.post(
            url=self._url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            json={"inputs": input},
        )
        response.raise_for_status()
        return response.json()
