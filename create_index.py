import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Загрузка данных
CSV_PATH = r"C:\Users\79672\Desktop\Diploma\Corpus.xlsx"
df = pd.read_excel(CSV_PATH)

# Проверяем, есть ли колонка 'answer', если нет - берем первую колонку
column_name = 'answer' if 'answer' in df.columns else df.columns[0]
chunks = df[column_name].astype(str).tolist()

# 2. Инициализация эмбеддингов
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 3. Создание базы
db = Chroma.from_texts(
    texts=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print(f"База создана из {len(chunks)} фрагментов и сохранена в /chroma_db")