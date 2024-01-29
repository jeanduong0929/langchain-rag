from chroma_client import ChromaClient
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
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

    combine_prompt_template = """
    Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). If you don't know the answer, just say that you don't know. Don't try to make up an answer. ALWAYS return a "SOURCES" part in your answer.
    QUESTION: {question}

    FINAL ANSWER:"""
    prompt = PromptTemplate.from_template(combine_prompt_template)
    question_generator_chain = LLMChain(llm=openai_llm, prompt=prompt)
    doc_chain = load_qa_chain(llm=openai_llm, chain_type="stuff")

    conv_retrieval_chain = ConversationalRetrievalChain(
        question_generator=question_generator_chain,
        memory=memory,
        retriever=chroma_db.vector_store.as_retriever(search_kwargs={"k": 5}),
        combine_docs_chain=doc_chain,
        return_source_documents=True,
    )

    # conv_retrieval_chain = ConversationalRetrievalChain.from_llm(
    #     llm=openai_llm,
    #     retriever=chroma_db.vector_store.as_retriever(search_kwargs={"k": 5}),
    #     chain_type="stuff",
    #     memory=memory,
    #     return_source_documents=True,
    #     verbose=False,
    #     condense_question_prompt=prompt,
    # )

    while True:
        # Clear the terminal
        clear()

        # Get user input
        query = get_user_input()

        # Quit if user types exit
        if query == "exit":
            break

        result = conv_retrieval_chain.run(question=query, output_key="answer")

        print("\n\n")
        print(result)

        press_enter()


if __name__ == "__main__":
    main()
