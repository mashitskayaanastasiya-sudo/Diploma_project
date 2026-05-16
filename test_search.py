from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # Обновили импорт, чтобы не было Warning

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

print("\n--- Система готова! Введите вопрос или 'exit' для выхода ---")

while True:
    query = input("\nВаш вопрос: ")
    
    if query.lower() == 'exit':
        break
        
    if not query.strip():
        continue

    # Ищем топ-3 подходящих куска
    docs = db.similarity_search(query, k=3)

    print(f"\nНайдено совпадений: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"\n--- Фрагмент №{i+1} ---")
        print(doc.page_content) 
        print("-" * 50)