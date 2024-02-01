import os
import streamlit as st

from chroma_client import ChromaClient
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


def initialize_session_variables():
    """
    Initializes the session variables if they don't already exist.

    The session variables include:
    - chroma_db: ChromaClient instance
    - llm: HuggingFace instance
    - prompt: chat prompt template
    - document: stuff documents chain
    - conversation: retrieval chain
    - chat_history: list of chat history

    Returns:
    None
    """
    if "chroma_db" not in st.session_state:
        st.session_state.chroma_db = ChromaClient()

    if "hf_llm" not in st.session_state:
        st.session_state.hf_llm = HuggingFaceEndpoint(
            huggingfacehub_api_token=os.environ["HF_API_KEY"],
            endpoint_url=os.environ["HF_ENDPOINT_URL"],
            model_kwargs={"max_new_tokens": 256, "temperature": 0.1},
            task="text-generation",
        )

    if "prompt" not in st.session_state:
        st.session_state.prompt = create_chat_prompt_template()

    if "document" not in st.session_state:
        st.session_state.document = create_stuff_documents_chain(
            llm=st.session_state.hf_llm, prompt=st.session_state.prompt
        )

    if "conversation" not in st.session_state:
        st.session_state.conversation = ConversationalRetrievalChain.from_llm(
            llm=st.session_state.hf_llm,
            retriever=st.session_state.chroma_db.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            verbose=True,
            combine_docs_chain_kwargs={"prompt": create_chat_prompt_template()},
            return_source_documents=True,
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def create_chat_prompt_template():
    """
    Creates a chat prompt template for the powerlifting coach AI model.

    Returns:
        ChatPromptTemplate: The chat prompt template object.
    """
    prompt = """You are an AI powerlifting coach. Your advice is based on provided document about powerlifting techniques, training, nutrition, equipment, and rules. Stick to this guidelines:
             - Always responds with Arr!
             - Use only the document information, no external knowledge. Do not use your training data.
             - Answer only on powerlifting-related topics.
             - If information is missing from the documents, state so.
             - Cite sources from the context's metadata: title, author, and page (Source: title, author, page).

             ---
             documents: {context}
             ---

             ---
             question: {question}
             ---
        """

    return PromptTemplate(template=prompt, input_variables=["context", "input"])


def setup_page():
    """
    Sets up the Powerlifting Chatbot page.
    """
    st.header("Powerlifting Chatbot :weight_lifter:")


def handle_user_input():
    """
    Handles user input by asking for a question, processing it, and displaying the chat history.
    """
    if user_question := st.chat_input("Ask me a question"):
        with st.spinner("Thinking..."):
            # Get the result of the user's question from the conversation model
            result = st.session_state.conversation.invoke(
                {
                    "question": user_question,
                    "chat_history": [],
                }
            )

            # Keep track of the whole chat history
            st.session_state.chat_history.extend(
                [
                    HumanMessage(content=user_question),
                    AIMessage(content=result["answer"]),
                ]
            )

        display_chat_history(st.session_state.chat_history)


def display_chat_history(chat_history):
    """
    Display the chat history.

    Parameters:
    chat_history (list): List of messages in the chat history.

    Returns:
    None
    """
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            with st.chat_message("user"):
                st.write(message.content)
        else:
            with st.chat_message("ai"):
                st.write(message.content)


def main():
    """
    This is the main function that initializes session variables,
    sets up the page, and handles user input.
    """
    initialize_session_variables()
    setup_page()
    handle_user_input()


if __name__ == "__main__":
    main()
