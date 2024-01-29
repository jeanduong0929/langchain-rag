from chroma_client import ChromaClient
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from utility import clear, press_enter


def get_user_input():
    return input(
        "Hello I am a powerlifting chatbot. Ask me a question (exit to quit): "
    ).lower()


def main():
    # Intialize the vector store
    chroma_db = ChromaClient()

    # Initialize OpenAI LLM
    openai_llm = ChatOpenAI(model="gpt-4", temperature=0.0)

    # Create a conversation buffer memory to save chat history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    prompt_template = """
    You are an AI model acting as a powerlifting coach. Your knowledge and advice are based entirely on a specific set of documents provided to you, referred to as {context}. These documents contain information about powerlifting techniques, training programs, nutrition, equipment, and competition rules. You do not have access to any other information beyond these documents. Your responses should be grounded solely in the information found in these documents. Here are your guidelines:

    1. Respond to inquiries using only the information contained in the {context}. Do not use your training data or external knowledge.
    2. If asked about powerlifting techniques, training routines, nutritional advice, or equipment, refer directly to the information in the provided documents to answer.
    3. In cases where the {context} does not contain the information needed to answer a query, clearly state that the answer is not available in the provided documents.
    4. Do not make assumptions or create answers based on general knowledge. Stick strictly to the content of the {context}.
    5. Maintain a professional tone, befitting a powerlifting coach, focusing on providing accurate and reliable information to assist in training and competition preparation.

    Your primary role is to assist, inform, and guide individuals interested in powerlifting by utilizing the specific information provided in the {context} documents.

    Question: {question}
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    conv_retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=openai_llm,
        retriever=chroma_db.vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        memory=memory,
        return_source_documents=True,
        verbose=False,
        # combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    while True:
        print("\n============================\n")

        # Get user input
        query = get_user_input()

        # Quit if user types exit
        if query == "exit":
            break

        result = conv_retrieval_chain.invoke({"question": query})

        print("\n")
        print(result["answer"])
        print("\n")
        print(result)


if __name__ == "__main__":
    main()
