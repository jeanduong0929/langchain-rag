import streamlit as st

from chroma_client import ChromaClient
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory


def initialize_session_variables():
    """
    Initializes the session variables if they don't already exist.

    The session variables include:
    - chroma_db: ChromaClient instance
    - openai_llm: ChatOpenAI instance
    - prompt: chat prompt template
    - document: stuff documents chain
    - conversation: retrieval chain
    - chat_history: list of chat history

    Returns:
    None
    """
    if "chroma_db" not in st.session_state:
        st.session_state.chroma_db = ChromaClient()

    if "openai_llm" not in st.session_state:
        st.session_state.openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    if "prompt" not in st.session_state:
        st.session_state.prompt = create_chat_prompt_template()

    if "document" not in st.session_state:
        st.session_state.document = create_stuff_documents_chain(
            llm=st.session_state.openai_llm, prompt=st.session_state.prompt
        )

    if "conversation" not in st.session_state:
        st.session_state.conversation = create_retrieval_chain(
            retriever=st.session_state.chroma_db.vector_store.as_retriever(
                search_kwargs={"k": 3}
            ),
            combine_docs_chain=st.session_state.document,
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def create_chat_prompt_template():
    """
    Creates a chat prompt template for the powerlifting coach AI model.

    Returns:
        ChatPromptTemplate: The chat prompt template object.
    """
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI powerlifting coach. Your advice is based on provided documents about powerlifting techniques, training, nutrition, equipment, and rules. Stick to this context:
                    - Use only the document information, no external knowledge.
                    - Answer only on powerlifting-related topics.
                    - If information is missing from the documents, state so.
                    - Maintain a professional tone.
                    - Cite sources from the context's metadata: title, author, and page (Source: title, author, page).
                    Reminder: I am an AI specialized in powerlifting, providing information based on specific documents. Please keep questions relevant to powerlifting.

                    ```
                    context: {context}
                    ```
                """,
            ),
            ("human", "{input}"),
        ]
    )


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
            chat_history = process_user_question(user_question)
        display_chat_history(chat_history)


def process_user_question(user_question):
    """
    Process the user's question by invoking the conversation model and updating the chat history.

    Args:
        user_question (str): The user's question.

    Returns:
        dict: The result of the conversation model invocation.
    """
    # Get the result of the user's question from the conversation model
    result = st.session_state.conversation.invoke(
        {
            "input": user_question,
        }
    )

    # Keep track of the whole chat history
    st.session_state.chat_history.extend(
        [HumanMessage(content=user_question), AIMessage(content=result["answer"])]
    )

    return st.session_state.chat_history


def load_chat_history(chat_history):
    """
    Load the chat history into the chat memory.

    Args:
        chat_history (list): List of messages in the chat history.

    Returns:
        None
    """
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            st.session_state.memory.chat_memory.add_user_message(message)
        else:
            st.session_state.memory.chat_memory.add_ai_message(message)


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
