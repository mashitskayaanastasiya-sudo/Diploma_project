import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Настройки путей
# Используем 'r' для корректной работы путей в Windows
PATH_TO_ARTICLES = r"C:\Users\79672\Desktop\Diploma\Articles" 
DB_DIR = "./chroma_db"

def run_ingestion():
    # 2. Инициализация инструментов
    # Параметры чанков для методологии (п. 2.a.ii)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    
    # Бесплатная модель эмбеддингов
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_chunks = []

    # 3. Чтение 62 статей
    print(f"Начинаю обработку файлов в {PATH_TO_ARTICLES}...")
    for filename in os.listdir(PATH_TO_ARTICLES):
        if filename.endswith(".txt"):
            file_path = os.path.join(PATH_TO_ARTICLES, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                chunks = text_splitter.split_text(text)
                all_chunks.extend(chunks)
                print(f"Добавлено {len(chunks)} чанков из файла: {filename}")

    # 4. Создание векторной базы
    if all_chunks:
        print(f"\nВсего создано {len(all_chunks)} чанков. Начинаю векторизацию...")
        # Это создаст папку chroma_db и сохранит там твои данные
        db = Chroma.from_texts(
            texts=all_chunks, 
            embedding=embeddings, 
            persist_directory=DB_DIR
        )
        print(f"Успех! Векторная база сохранена в {DB_DIR}")
    else:
        print("Ошибка: чанки не созданы. Проверь путь к папке и содержимое файлов.")

if __name__ == "__main__":
    run_ingestion()