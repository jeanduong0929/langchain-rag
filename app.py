from chroma_client import ChromaClient
from langchain_openai import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage


def get_user_input():
    return input(
        "Hello I am a powerlifting chatbot. Ask me a question (exit to quit): "
    ).lower()


def main():
    # Intialize the vector store
    chroma_db = ChromaClient()

    # Initialize OpenAI LLM
    openai_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI model acting as a powerlifting coach. Your knowledge and advice are based entirely on a specific set of documents provided to you, referred to as 'context'. These documents contain information about powerlifting techniques, training programs, nutrition, equipment, and competition rules. You do not have access to any other information beyond these documents. Your responses should be grounded solely in the information found in these documents. Here are your guidelines:
                1. Respond to inquiries using only the information contained in the context. Do not use your training data or external knowledge.
                2. If asked about powerlifting techniques, training routines, nutritional advice, or equipment, refer directly to the information in the provided documents to answer.
                3. In cases where the context does not contain the information needed to answer a query, clearly state that the answer is not available in the provided documents.
                4. Do not make assumptions or create answers based on general knowledge. Stick strictly to the content of the context.
                5. Maintain a professional tone, befitting a powerlifting coach, focusing on providing accurate and reliable information to assist in training and competition preparation.
                6. Quote the source from the context metadata
    
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

    document = create_stuff_documents_chain(llm=openai_llm, prompt=prompt)

    conv_retrieval_chain = create_retrieval_chain(
        retriever=chroma_db.vector_store.as_retriever(search_kwargs={"k": 3}),
        combine_docs_chain=document,
    )

    chat_history = []
    while True:
        print("\n============================\n")

        # Get user input
        query = get_user_input()

        # Quit if user types exit
        if query == "exit":
            break

        # Query the user input
        result = conv_retrieval_chain.invoke(
            {
                "chat_history": chat_history,
                "input": query,
            }
        )

        # Add the user input and response to the chat history
        chat_history.extend(
            [HumanMessage(content=query), AIMessage(content=result["answer"])]
        )

        print("\n")
        print(result["answer"])


if __name__ == "__main__":
    main()
