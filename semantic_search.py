import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd

def get_vector_db(model_path, db_path, csv_path):
    embeddings = HuggingFaceEmbeddings(model_name=model_path)
    
    if not os.path.exists(db_path):
        # Логика создания базы (индексация)
        df = pd.read_excel(csv_path, header=1)
        documents = []
        for _, row in df.iterrows():
            q = str(row.get('question', ''))
            a = str(row.get('answer', ''))
            if q.strip() and a.strip():
                # Мы сохраняем только ответ в контент, как ты и хотела
                doc = Document(
                    page_content=a,
                    metadata={"source": str(row.get('source_article', 'Corpus')), "question_ref": q}
                )
                documents.append(doc)
        
        db = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory=db_path)
    else:
        # Подключение к существующей базе
        db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    return db