import streamlit as st

from chroma_client import ChromaClient
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage


def main():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Intialize the vector store
    st.session_state.chroma_db = ChromaClient()

    # Initialize OpenAI LLM
    st.session_state.openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # Create the prompt for the chatbot
    st.session_state.prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI model acting as a powerlifting coach. Your knowledge and advice are based entirely on a specific set of documents provided to you, referred to as 'context'. These documents contain information about powerlifting techniques, training programs, nutrition, equipment, and competition rules. You do not have access to any other information beyond these documents. Your responses should be grounded solely in the information found in these documents. Here are your guidelines:
                1. Respond to inquiries using only the information contained in the context. Do not use your training data or external knowledge.
                2. If asked about powerlifting techniques, training routines, nutritional advice, or equipment, refer directly to the information in the provided documents to answer.
                3. In cases where the context does not contain the information needed to answer a query, clearly state that the answer is not available in the provided documents.
                4. Do not make assumptions or create answers based on general knowledge. Stick strictly to the content of the context.
                5. Maintain a professional tone, befitting a powerlifting coach, focusing on providing accurate and reliable information to assist in training and competition preparation.
                6. Quote the source from the context metadata. The source should contain the name of the author, the title of the document, and page number.

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

    st.session_state.document = create_stuff_documents_chain(
        llm=st.session_state.openai_llm, prompt=st.session_state.prompt
    )

    st.session_state.conversation = create_retrieval_chain(
        retriever=st.session_state.chroma_db.vector_store.as_retriever(
            search_kwargs={"k": 3}
        ),
        combine_docs_chain=st.session_state.document,
    )

    # Setting up the page title and icon
    st.set_page_config(page_title="Powerlifting Chatbot", page_icon=":weight_lifter:")

    # Setting up the header
    st.header("Powerlifting Chatbot :weight_lifter:")

    if user_question := st.chat_input("Ask me a question"):
        # Query the user input
        with st.spinner("Thinking..."):
            result = st.session_state.conversation.invoke(
                {"chat_history": st.session_state.chat_history, "input": user_question}
            )

        # Save the chat history to the session state
        st.session_state.chat_history = result["chat_history"]

        # Add the user input and response to the chat history
        st.session_state.chat_history.extend(
            [HumanMessage(content=user_question), AIMessage(content=result["answer"])]
        )

        # Iterate through the chat history and display the messages
        st.session_state.chat_history = result["chat_history"]
        for i, message in enumerate(result["chat_history"]):
            if i % 2 == 0:
                with st.chat_message("user"):
                    st.write(message.content)
            else:
                with st.chat_message("ai"):
                    st.write(message.content)


if __name__ == "__main__":
    main()
