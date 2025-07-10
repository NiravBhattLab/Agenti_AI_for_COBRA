import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Metabolic Model Assistant", layout="wide")

st.markdown(
    """
    <style>
    div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "model_id" not in st.session_state:
    st.session_state.model_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "disclaimer" not in st.session_state:
    st.session_state.disclaimer = False

@st.dialog("DISCLAIMER!")
def popup():
    st.write(f"This app assumes that the user has prior knowledge about Constraint Based Metabolic Models (CBBMs)")
    st.html("<span class='big-dialog'></span>")
    if st.button("Proceed"):
        st.session_state.disclaimer = True
        st.rerun()

if not st.session_state.disclaimer:
    popup()
else:
    with st.sidebar:
        st.header("🧠 LLM Configuration")

        provider = st.selectbox("Choose LLM Provider", ["ollama", "openai", "groq", "Hugging Face"])
        model = st.text_input("Model Name", value="llama3.1:latest" if provider == "ollama" else "")
        api_key = None

        if provider == "openai":
            api_key = st.text_input("Enter API Key", type="password")
        elif provider == "Hugging Face":
            api_key = st.text_input("Enter HF TOKEN", type="password")
        elif provider == "groq":
            api_key = st.text_input("Enter API Key", type="password")

        if st.button("Apply LLM"):
            payload = {"provider": provider, "model": model}
            if provider != "ollama":
                payload["api_key"] = api_key
            res = requests.post(f"{API_BASE}/set_llm/", json=payload)
            if res.status_code == 200:
                st.success(f"LLM switched to {provider}: {model}")
            else:
                st.error(f"Failed to switch LLM: {res.json().get('detail')}")


        st.header("📁 Model Management")

        csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if csv_file and st.button("Upload CSV to Backend"):
            files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
            res = requests.post(f"{API_BASE}/upload_csv/", files=files)
            if res.status_code == 200 and res.json().get("status") == "success":
                st.success(f"CSV uploaded: {res.json()['filename']}")
            else:
                st.error(f"Upload failed: {res.json().get('detail')}")

        uploaded_file = st.file_uploader("Upload SBML (.xml) file", type=["xml"])
        if uploaded_file and st.button("Upload Model", key="upload_btn"):
            files = {"file": uploaded_file}
            res = requests.post(f"{API_BASE}/upload_model/", files=files)
            if res.status_code == 200:
                st.session_state.model_id = res.json()["model_id"]
                st.success(f"Model ID: {st.session_state.model_id}")
            else:
                st.error(f"Upload failed: {res.json().get('detail')}")

        if st.session_state.model_id:
            if st.button("📊 Get Model Stats", key="stats_btn"):
                res = requests.get(f"{API_BASE}/get_stats/")
                if res.status_code == 200:
                    st.info(res.json()["stats"])
                else:
                    st.error("Failed to retrieve model stats")

    st.header("💬 Agentic Chat")

    chat_container = st.container()
    user_input = st.chat_input("Ask about the model...")

    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        try:
            res = requests.post(f"{API_BASE}/chat/", json={"message": user_input})
            if res.status_code == 200:
                response_text = res.json()["response"]  # ["raw"]["message"]["content"] # Here if OpenAI then chnage accordingly
                st.session_state.chat_history.append(("agent", response_text))
            else:
                error_msg = res.json().get("detail", "Unknown error")
                st.session_state.chat_history.append(("agent", f"⚠️ Error: {error_msg}"))
        except Exception as e:
            st.session_state.chat_history.append(("agent", f"⚠️ Exception: {str(e)}"))

    with chat_container:
        for role, msg in st.session_state.chat_history:
            if role == "user":
                st.chat_message("user").write(msg)
            else:
                st.chat_message("assistant").write(msg)