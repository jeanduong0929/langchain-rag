import streamlit as st

from chroma_client import ChromaClient
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage


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
                """You are an AI model acting as a powerlifting coach. Your knowledge and advice are based entirely on a specific set of documents provided to you, referred to as 'context'. These documents contain information about powerlifting techniques, training programs, nutrition, equipment, and competition rules. You do not have access to any other information beyond these documents. Your responses should be grounded solely in the information found in these documents. Here are your guidelines:
                    1. Respond to inquiries using only the information contained in the context. Do not use your training data or external knowledge.
                    2. If asked about powerlifting techniques, training routines, nutritional advice, or equipment, refer directly to the information in the provided documents to answer.
                    3. In cases where the context does not contain the information needed to answer a query, clearly state that the answer is not available in the provided documents.
                    4. Do not make assumptions or create answers based on general knowledge. Stick strictly to the content of the context.
                    5. Maintain a professional tone, befitting a powerlifting coach, focusing on providing accurate and reliable information to assist in training and competition preparation.
                    6. Quote the sources from the context metadata. The source should contain the name of the author, the title of the document, and page number.

                    Your primary role is to assist, inform, and guide individuals interested in powerlifting by utilizing the specific information provided in the context documents.

                    ---
                    Context: {context}
                    ---
                    """,
            ),
            (MessagesPlaceholder(variable_name="chat_history")),
            ("user", "Question: {input}"),
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
            result = process_user_question(user_question)
        display_chat_history(result)


def process_user_question(user_question):
    """
    Process the user's question by invoking the conversation model and updating the chat history.

    Args:
        user_question (str): The user's question.

    Returns:
        dict: The result of the conversation model invocation.
    """
    result = st.session_state.conversation.invoke(
        {"chat_history": st.session_state.chat_history, "input": user_question}
    )
    st.session_state.chat_history = result["chat_history"]
    st.session_state.chat_history.extend(
        [HumanMessage(content=user_question), AIMessage(content=result["answer"])]
    )
    return result


def display_chat_history(result):
    """
    Display the chat history.

    Parameters:
    result (dict): The result containing the chat history.

    Returns:
    None
    """
    for i, message in enumerate(result["chat_history"]):
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
