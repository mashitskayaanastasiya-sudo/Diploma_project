import os

from langchain_aws import ChatBedrock
from langchain_classic.chains import RetrievalQA


def ask_bedrock(query, vector_db):
    required_env_vars = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]
    if any(not os.getenv(env_var) for env_var in required_env_vars):
        return "Model configuration error."

    llm = ChatBedrock(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        provider="anthropic",
        model_kwargs={
            "temperature": 0
        }
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
        return f"Bedrock Error: {e}"
