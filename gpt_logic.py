import os

from langchain_openai import ChatOpenAI
from langchain_classic.chains import RetrievalQA

def ask_gpt(query, vector_db):
    if not os.getenv("OPENAI_API_KEY"):
        return "Model configuration error."

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 3})
    )

    try:
        response = qa_chain.invoke({"query": query})
        return response["result"]

    except Exception as e:
        return f"GPT Error: {e}"
