import os
import re

def run_regexp_search(query, articles_path="./Articles"):
    """
    Выполняет поиск по фразе и подсвечивает совпадения.
    """
    matches = []
    if not os.path.exists(articles_path):
        return None, "Directory not found. Please create 'Articles' folder."

    clean_query = query.strip()
    if not clean_query:
        return [], "Empty query"

    # Создаем паттерн для поиска и подсветки
    search_pattern = re.compile(re.escape(clean_query), re.IGNORECASE)
    
    for file_name in os.listdir(articles_path):
        if file_name.endswith(".txt"):
            file_full_path = os.path.join(articles_path, file_name)
            try:
                with open(file_full_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    if re.search(search_pattern, text):
                        # Функция для замены: берет найденный текст и оборачивает в <mark>
                        # Это сохраняет оригинальный регистр букв в тексте
                        highlighted_text = re.sub(
                            search_pattern, 
                            lambda m: f"<mark style='background-color: #FFFF00; color: black;'>{m.group()}</mark>", 
                            text
                        )
                        
                        matches.append({
                            "file": file_name, 
                            "content": highlighted_text
                        })
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
    
    return matches, None