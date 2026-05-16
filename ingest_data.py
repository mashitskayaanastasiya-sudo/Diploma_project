import chromadb
import os

# 1. Инициализация базы
# Все данные ChromaDB сохранит в папку 'compliance_db' внутри папки 'Diploma'
client = chromadb.PersistentClient(path="./compliance_db")
collection = client.get_or_create_collection(name="compliance_articles")

# 2. Путь к твоим статьям
articles_path = "./Articles"

# 3. Цикл загрузки
documents = []
metadatas = []
ids = []

for filename in os.listdir(articles_path):
    if filename.endswith(".txt"):
        with open(os.path.join(articles_path, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            documents.append(content)
            # Сохраняем имя файла в метаданные, чтобы знать, откуда пришел ответ
            metadatas.append({"source": filename})
            ids.append(filename)

# 4. Добавляем всё в ChromaDB
if documents:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    print(f"Успешно загружено {len(documents)} статей в коллекцию.")