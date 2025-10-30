from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.retriever import MilvusRetriever
from llm_selector import get_llm
from utils.prompts import prompt

llm = get_llm(provider="groq")
retriever = MilvusRetriever(
        collection_name="documents_chunks",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        milvus_host="localhost",
        milvus_port="19530"
    )

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "who is syed saleem.?"})
print(response["answer"])