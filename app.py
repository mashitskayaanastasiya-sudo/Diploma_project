import streamlit as st
import csv
import json
import os
import sys
import time
import tomllib
import tempfile
import hashlib
from datetime import datetime, timezone

# --- ЖЕСТКАЯ ПРИВЯЗКА ПУТИ ---
current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- CONFIGURATION ---
CSV_PATH = os.path.join(current_dir, "Corpus.xlsx")

TUNED_MODEL_PATH = os.path.join(
    current_dir,
    "my_tuned_model-20260515T121335Z-3-001",
    "my_tuned_model"
)

def get_corpus_version(path):
    stat = os.stat(path)
    version_source = f"{stat.st_size}-{int(stat.st_mtime)}".encode("utf-8")
    return hashlib.md5(version_source).hexdigest()[:8]


CORPUS_VERSION = get_corpus_version(CSV_PATH)
CHROMA_TUNED_PATH = os.path.join(
    tempfile.gettempdir(), f"compliance_chroma_db_tuned_{CORPUS_VERSION}"
)
CHROMA_BASE_PATH = os.path.join(
    tempfile.gettempdir(), f"compliance_chroma_db_base_{CORPUS_VERSION}"
)
LOGS_DIR = os.path.join(current_dir, "logs")
QUERY_LOG_PATH = os.path.join(LOGS_DIR, "query_logs.csv")

st.set_page_config(page_title="Compliance AI Assistant", layout="wide")

st.markdown(
    """
    <style>
        [data-testid="stToolbar"],
        [data-testid="stMainMenu"],
        [data-testid="stDeployButton"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

from baseline_search import run_regexp_search
from semantic_search import get_vector_db


LOCAL_SECRETS_CACHE = None


def get_local_secrets():
    global LOCAL_SECRETS_CACHE

    if LOCAL_SECRETS_CACHE is not None:
        return LOCAL_SECRETS_CACHE

    secrets_path = os.path.join(current_dir, ".streamlit", "secrets.toml")
    if not os.path.exists(secrets_path):
        LOCAL_SECRETS_CACHE = {}
        return LOCAL_SECRETS_CACHE

    with open(secrets_path, "rb") as secrets_file:
        LOCAL_SECRETS_CACHE = tomllib.load(secrets_file)

    return LOCAL_SECRETS_CACHE


def get_secret_value(section, key, default=None):
    try:
        value = st.secrets.get(section, {}).get(key)
        if value is not None:
            return value
    except Exception:
        pass

    return get_local_secrets().get(section, {}).get(key, default)


def parse_user_ids(value):
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in value if str(item).strip()]


def get_auth_config():
    admin_id = get_secret_value("auth", "admin_id") or os.getenv("APP_ADMIN_ID", "")
    user_ids = parse_user_ids(
        get_secret_value("auth", "user_ids") or os.getenv("APP_USER_IDS", "")
    )

    admin_id = admin_id.strip()
    allowed_ids = set(user_ids)
    if admin_id:
        allowed_ids.add(admin_id)

    return admin_id, allowed_ids


def configure_environment_secrets():
    secret_to_env = {
        "google_api_key": "GOOGLE_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
        "deepseek_api_key": "DEEPSEEK_API_KEY",
        "aws_access_key_id": "AWS_ACCESS_KEY_ID",
        "aws_secret_access_key": "AWS_SECRET_ACCESS_KEY",
        "aws_default_region": "AWS_DEFAULT_REGION",
    }

    for secret_name, env_name in secret_to_env.items():
        value = get_secret_value("api_keys", secret_name)
        if value and not os.getenv(env_name):
            os.environ[env_name] = str(value)


def require_authorized_user():
    admin_id, allowed_ids = get_auth_config()

    if "user_id" not in st.session_state:
        st.session_state.user_id = None

    if not allowed_ids:
        st.error("Authorization is not configured. Please add tester IDs to app secrets.")
        st.stop()

    if st.session_state.user_id in allowed_ids:
        return st.session_state.user_id, st.session_state.user_id == admin_id

    st.title("Compliance Model Testing")
    entered_id = st.text_input("Enter your tester ID:", type="password")

    if st.button("Sign in"):
        entered_id = entered_id.strip()
        if entered_id in allowed_ids:
            st.session_state.user_id = entered_id
            st.rerun()
        else:
            st.error("Invalid tester ID.")

    st.stop()


def log_query(user_id, selected_model, internal_model, query, response_time):
    os.makedirs(LOGS_DIR, exist_ok=True)
    file_exists = os.path.exists(QUERY_LOG_PATH)

    with open(QUERY_LOG_PATH, "a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "timestamp_utc",
                "user_id",
                "model_label",
                "internal_model",
                "query",
                "response_time_sec",
            ],
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "model_label": selected_model,
                "internal_model": internal_model,
                "query": query,
                "response_time_sec": round(response_time, 4),
            }
        )


def show_model_error(selected_model, error, is_admin):
    st.error(f"{selected_model} error. Please try again or contact the administrator.")
    if is_admin:
        st.exception(error)


configure_environment_secrets()


try:
    import gemini_logic
except ModuleNotFoundError as e:
    gemini_logic = None
    st.error("A model component failed to load.")

try:
    import gpt_logic
except ModuleNotFoundError as e:
    gpt_logic = None
    st.error("A model component failed to load.")

try:
    import bedrock_logic
except ModuleNotFoundError as e:
    bedrock_logic = None
    st.error("A model component failed to load.")

try:
    import deepseek_logic
except ModuleNotFoundError as e:
    deepseek_logic = None
    st.error("A model component failed to load.")

# --- SIDEBAR ---
MODEL_LABELS = {
    "Model 1": "Gemini RAG (Assistant)",
    "Model 2": "GPT RAG (Assistant)",
    "Model 3": "Bedrock RAG (Assistant)",
    "Model 4": "DeepSeek RAG (Assistant)",
    "Model 5": "Semantic Search (Fine-tuned)",
    "Model 6": "Semantic Search (Base)",
    "Model 7": "Regexp (Baseline)",
}

user_id, is_admin = require_authorized_user()

selected_model = st.sidebar.radio("Select Model:", tuple(MODEL_LABELS.keys()))
model_choice = MODEL_LABELS[selected_model]

# --- MAIN INTERFACE ---
st.title("🔎 AI-Powered Compliance Search")
with st.form("query_form"):
    query = st.text_input("Enter your query:")
    submitted = st.form_submit_button("Submit")

if submitted and query:
    start_time = time.time()

    if model_choice == "Regexp (Baseline)":
        with st.spinner(f"{selected_model} is thinking..."):
            results, error = run_regexp_search(query)

            if error:
                st.error(f"Error: {error}")
            elif results:
                end_time = time.time()
                log_query(user_id, selected_model, model_choice, query, end_time - start_time)
                st.success("Model Response:")
                for index, m in enumerate(results, start=1):
                    with st.expander(f"Result {index}"):
                        st.markdown(m["content"], unsafe_allow_html=True)
            else:
                st.warning("No exact matches found.")

    elif model_choice == "Gemini RAG (Assistant)":
        if gemini_logic is None:
            st.error(f"{selected_model} is unavailable.")
        else:
            with st.spinner(f"{selected_model} is thinking..."):
                try:
                    db = get_vector_db(TUNED_MODEL_PATH, CHROMA_TUNED_PATH, CSV_PATH)
                    answer = gemini_logic.ask_gemini(query, db)
                    end_time = time.time()
                    log_query(user_id, selected_model, model_choice, query, end_time - start_time)

                    st.success("Model Response:")
                    st.markdown(answer)
                    st.caption(f"**Response time:** {round(end_time - start_time, 4)} sec.")

                except Exception as e:
                    show_model_error(selected_model, e, is_admin)

    elif model_choice == "GPT RAG (Assistant)":
        if gpt_logic is None:
            st.error(f"{selected_model} is unavailable.")
        else:
            with st.spinner(f"{selected_model} is thinking..."):
                try:
                    db = get_vector_db(TUNED_MODEL_PATH, CHROMA_TUNED_PATH, CSV_PATH)
                    answer = gpt_logic.ask_gpt(query, db)
                    end_time = time.time()
                    log_query(user_id, selected_model, model_choice, query, end_time - start_time)

                    st.success("Model Response:")
                    st.markdown(answer)
                    st.caption(f"**Response time:** {round(end_time - start_time, 4)} sec.")

                except Exception as e:
                    show_model_error(selected_model, e, is_admin)

    elif model_choice == "Bedrock RAG (Assistant)":
        if bedrock_logic is None:
            st.error(f"{selected_model} is unavailable.")
        else:
            with st.spinner(f"{selected_model} is thinking..."):
                try:
                    db = get_vector_db(TUNED_MODEL_PATH, CHROMA_TUNED_PATH, CSV_PATH)
                    answer = bedrock_logic.ask_bedrock(query, db)
                    end_time = time.time()
                    log_query(user_id, selected_model, model_choice, query, end_time - start_time)

                    st.success("Model Response:")
                    st.markdown(answer)
                    st.caption(f"**Response time:** {round(end_time - start_time, 4)} sec.")

                except Exception as e:
                    show_model_error(selected_model, e, is_admin)

    elif model_choice == "DeepSeek RAG (Assistant)":
        if deepseek_logic is None:
            st.error(f"{selected_model} is unavailable.")
        else:
            with st.spinner(f"{selected_model} is thinking..."):
                try:
                    db = get_vector_db(TUNED_MODEL_PATH, CHROMA_TUNED_PATH, CSV_PATH)
                    answer = deepseek_logic.ask_deepseek(query, db)
                    end_time = time.time()
                    log_query(user_id, selected_model, model_choice, query, end_time - start_time)

                    st.success("Model Response:")
                    st.markdown(answer)
                    st.caption(f"**Response time:** {round(end_time - start_time, 4)} sec.")

                except Exception as e:
                    show_model_error(selected_model, e, is_admin)

    else:
        if model_choice == "Semantic Search (Fine-tuned)":
            m_path = TUNED_MODEL_PATH
            d_path = CHROMA_TUNED_PATH
        else:
            m_path = "all-MiniLM-L6-v2"
            d_path = CHROMA_BASE_PATH

        with st.spinner(f"{selected_model} is thinking..."):
            try:
                db = get_vector_db(m_path, d_path, CSV_PATH)
                search_results = db.similarity_search(query, k=1)

                if search_results:
                    end_time = time.time()
                    log_query(user_id, selected_model, model_choice, query, end_time - start_time)

                    st.success("Model Response:")
                    st.markdown(search_results[0].page_content)

                    st.divider()
                    st.caption(f"**Response time:** {round(end_time - start_time, 4)} sec.")
                else:
                    st.error("No relevant information found.")

            except Exception as e:
                show_model_error(selected_model, e, is_admin)

# --- FOOTER ---
st.sidebar.divider()

if st.sidebar.button("Clear DB Cache"):
    st.cache_resource.clear()
    st.sidebar.success("Cache cleared successfully.")

if is_admin:
    st.sidebar.divider()
    st.sidebar.caption(f"Signed in as admin: {user_id}")
    if os.path.exists(QUERY_LOG_PATH):
        with open(QUERY_LOG_PATH, "rb") as log_file:
            log_data = log_file.read()
            st.sidebar.download_button(
                "Download query logs",
                data=log_data,
                file_name="query_logs.csv",
                mime="text/csv",
            )
    else:
        st.sidebar.caption("No query logs yet.")
else:
    st.sidebar.caption(f"Signed in as: {user_id}")
