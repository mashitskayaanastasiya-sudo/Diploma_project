import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Настройка сплиттера
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

# 2. Функция для чтения твоих анонимизированных статей
def process_articles(folder_path):
    all_chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
                text = f.read()
                # Разбиваем текст на части
                chunks = text_splitter.split_text(text)
                all_chunks.extend(chunks)
                print(f"Файл {filename} разбит на {len(chunks)} частей.")
    return all_chunks

# Запуск
path = r"C:\Users\79672\Desktop\Diploma\Articles" # путь к папке с 62 статьями
chunks = process_articles(path)
print(f"Всего создано {len(chunks)} чанков для Semantic Search.")
