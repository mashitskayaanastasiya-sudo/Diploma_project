import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI
from langchain_classic.chains import RetrievalQA
from langchain_core.language_models.llms import LLM
from typing import Optional, List, Any


class DeepSeekLLM(LLM):
    api_key: str
    model_name: str = "deepseek-v4-flash"

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        **kwargs: Any
    ) -> str:
        client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional compliance assistant. Answer only based on the provided context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        return response.choices[0].message.content


def ask_deepseek(query, vector_db):
    if not os.getenv("DEEPSEEK_API_KEY"):
        return "Model configuration error."

    llm = DeepSeekLLM(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model_name="deepseek-v4-flash"
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
        return f"DeepSeek Error: {e}"
