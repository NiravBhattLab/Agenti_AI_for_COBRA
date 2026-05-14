import streamlit as st
import requests
import os

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Metabolic Model Assistant", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1rem;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }
    div[data-testid="stDialog"] div[role="dialog"]:has(.big-dialog) {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }

    [data-testid="stHeader"] {
        background: transparent !important;
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
if "llm" not in st.session_state:
    st.session_state.llm = False
if "session_ready" not in st.session_state:
    st.session_state.session_ready = False
if "session_name" not in st.session_state:
    st.session_state.session_name = None

@st.dialog("🗂️ Name Your Session")
def name_session_dialog():
    st.markdown("Give this session a descriptive name. All artifacts (FVA, knockouts, sampling) will be saved in a folder with this name under `outputs/`.")
    name = st.text_input("Session name", placeholder="e.g., ecoli_growth_analysis")
    if st.button("Start Session", type="primary", use_container_width=True):
        if not name.strip():
            st.error("Please enter a session name.")
        else:
            try:
                res = requests.post(f"{API_BASE}/create_session/", json={"session_name": name.strip()})
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.session_name = data["session_name"]
                    st.session_state.session_ready = True
                    st.rerun()
                else:
                    st.error(f"Failed to create session: {res.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Could not reach backend: {e}")

@st.dialog("DISCLAIMER!")
def popup():
    st.write(f"This app assumes that the user has prior knowledge about Constraint Based Metabolic Models (CBBMs)")
    st.html("<span class='big-dialog'></span>")
    if st.button("Proceed"):
        st.session_state.disclaimer = True
        st.rerun()


@st.dialog("🧠 LLM Configuration", width="large")
def llm_configuration_dialog():
    st.markdown("Set or change the LLM Provider for Agent interaction.")
    
    provider = st.selectbox("Choose LLM Provider", ["groq", "ollama", "openai", "Hugging Face"])
    model = st.text_input("Model Name", value="llama-3.1-8b-instant" if provider == "groq" else "")
    api_key = None

    if provider == "openai":
        api_key = st.text_input("Enter API Key", type="password")
    elif provider == "Hugging Face":
        api_key = st.text_input("Enter HF TOKEN", type="password")
    elif provider == "groq":
        api_key = st.text_input("Enter API Key", type="password")

    if st.button("Apply LLM"):
        if provider not in ['groq', 'openai', 'Hugging Face'] or not api_key:
            st.error("API Key is required for Groq, OpenAI and Hugging Face providers.")
        else:
            payload = {"provider": provider, "model": model}
            if provider != "ollama":
                payload["api_key"] = api_key
            res = requests.post(f"{API_BASE}/set_llm/", json=payload)
            if res.status_code == 200:
                st.success(f"LLM switched to {provider}: {model}")
                st.session_state.llm = True
                st.session_state.llm_provider = provider
                st.session_state.llm_model = model
                st.rerun()
            else:
                st.error(f"Failed to switch LLM: {res.json().get('detail')}")

@st.dialog("⏹ End Session", width="large")
def end_session_dialog():
    st.markdown(f"**Current session:** `{st.session_state.session_name}`")
    st.markdown("Choose what to do with the artifacts (FVA results, knockouts, sampling CSVs) generated in this session.")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Artifacts", use_container_width=True, type="primary"):
            try:
                res = requests.post(f"{API_BASE}/end_session/", json={"keep": True})
                if res.status_code == 200:
                    st.success(f"Artifacts saved in `outputs/{st.session_state.session_name}`.")
            except Exception as e:
                st.error(f"Error: {e}")
    with col2:
        if st.button("🗑️ Delete Artifacts", use_container_width=True):
            try:
                res = requests.post(f"{API_BASE}/end_session/", json={"keep": False})
                if res.status_code == 200:
                    st.success("Session artifacts deleted.")
            except Exception as e:
                st.error(f"Error: {e}")
    st.divider()
    if st.button("🛑 Quit Application", use_container_width=True):
        try:
            requests.post(f"{API_BASE}/shutdown/", timeout=2)
        except Exception:
            pass
        os._exit(0)

@st.dialog("📁 Model Management", width="large")
def model_management_dialog():
    st.markdown("Use the tools below to manage models and settings.")

    uploaded_file = st.file_uploader("Upload SBML (.xml) file", type=["xml"])
    if uploaded_file and st.button("Upload Model", key="upload_btn"):
        files = {"file": uploaded_file}
        res = requests.post(f"{API_BASE}/upload_model/", files=files)
        if res.status_code == 200:
            st.session_state.model_id = res.json()["model_id"]
            st.success(f"Model ID: {st.session_state.model_id}")
        else:
            st.error(f"Upload failed: {res.json().get('detail')}")

    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if csv_file and st.button("Upload CSV to Backend"):
        files = {"file": (csv_file.name, csv_file.getvalue(), "text/csv")}
        res = requests.post(f"{API_BASE}/upload_csv/", files=files)
        if res.status_code == 200 and res.json().get("status") == "success":
            st.success(f"CSV uploaded: {res.json()['filename']}")
        else:
            st.error(f"Upload failed: {res.json().get('detail')}")

    if st.session_state.model_id:
        if st.button("📊 Get Model Stats", key="stats_btn"):
            res = requests.get(f"{API_BASE}/get_stats/")
            if res.status_code == 200:
                st.info(res.json()["stats"])
            else:
                st.error("Failed to retrieve model stats")

        st.markdown("#### 🎯 Set Objective Function")
        with st.form("objective_form", clear_on_submit=False):
            col1, col2 = st.columns([2, 1], vertical_alignment="center")
            with col1:
                obj_str = st.text_input("Objective expression",key="obj_reaction_input",placeholder="e.g., 1 ATPM + 2 EX_o2_e",help="Examples: `ATPM`, `1 ATPM + 2 EX_o2_e`, `1*ATPM + 2*EX_o2_e`, `0.5 ATPM - 3.25 EX_o2_e`")
            with col2:
                obj_direction = st.selectbox("Direction", ["max", "min"], index=0, key="obj_direction")
            submit_obj = st.form_submit_button("Set Objective", use_container_width=True)

        if submit_obj:
            if not obj_str or not obj_str.strip():
                st.error("Please enter a valid objective expression.")
            else:
                payload = {"objective_str": obj_str.strip(), "direction": obj_direction}
                try:
                    res = requests.post(f"{API_BASE}/set_objective/", json=payload, timeout=30)
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Objective set successfully.\n\n**Direction:** {data.get('direction')}\n\n**Expression:** `{data.get('objective')}`")
                    else:
                        detail = res.json().get("detail", f"HTTP {res.status_code}")
                        st.error(f"Failed to set objective: {detail}")
                except Exception as e:
                    st.error(f"Backend error: {e}")

        st.markdown("#### 🔁 Configure Sampler")
        method = st.selectbox("Sampler method",["optgp", "achr"],index=0,help="Choose OptGP (parallel, fast) or ACHR (classical hit-and-run).")
        if st.button("Apply Sampler", use_container_width=True, key="btn_apply_sampler"):
            payload = {"method": method}
            try:
                res = requests.post(f"{API_BASE}/set_sampler/", json=payload, timeout=30)
                if res.status_code == 200:
                    data = res.json()
                    s = data.get("sampler", "")
                    st.success(f"Sampler set: {s}")
                else:
                    st.error(res.json().get("detail", f"HTTP {res.status_code}"))
            except Exception as e:
                st.error(f"Failed to set sampler: {e}")


if not st.session_state.session_ready:
    name_session_dialog()
elif not st.session_state.disclaimer:
    popup()
else:
    col1, col2, col3, col4 = st.columns([91, 3, 3, 3])
    with col1:
        st.header("💬 Agentic Chat")
    with col2:
        if st.button("📁", help="Open Model Management", use_container_width=True): model_management_dialog()
    with col3:
        if st.button("🧠", help="Open LLM Configuration", use_container_width=True): llm_configuration_dialog()
    with col4:
        if st.button("⏹", help="End Session", use_container_width=True): end_session_dialog()

    if st.session_state.llm:
        sub = f"Provider: {st.session_state.get('llm_provider','?')} | Model: {st.session_state.get('llm_model','?')} | Session: {st.session_state.get('session_name','?')}"
        st.caption(sub)
        chat_container = st.container()
        user_input = st.chat_input("Ask about the model...")

        if user_input:
            st.session_state.chat_history.append(("user", user_input))
            try:
                res = requests.post(f"{API_BASE}/chat/", json={"message": user_input})
                if res.status_code == 200:
                    data = res.json()
                    response_text = data["response"]
                    if data.get("model_id"):
                        st.session_state.model_id = data["model_id"]
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
    else:
        st.warning("Please configure the LLM in the sidebar to start chatting.")
        st.info("Supported providers: groq, ollama, openai, Hugging Face")