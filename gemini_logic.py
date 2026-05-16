import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA

def ask_gemini(query, vector_db):
    if not os.getenv("GOOGLE_API_KEY"):
        return "Model configuration error."

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
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
        return f"Gemini Error: {e}"
