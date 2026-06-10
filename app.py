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

    /* Icon buttons in the header toolbar */
    div.toolbar-btn button {
        border-radius: 50% !important;
        width: 2.4rem !important;
        height: 2.4rem !important;
        min-width: 0 !important;
        padding: 0 !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        background: transparent !important;
        color: inherit !important;
        opacity: 0.7;
        transition: opacity 0.15s, background 0.15s, transform 0.1s !important;
        cursor: pointer !important;
    }
    div.toolbar-btn button:hover {
        background: rgba(255,255,255,0.1) !important;
        opacity: 1 !important;
        transform: scale(1.12) !important;
    }
    div.toolbar-btn button:active {
        transform: scale(0.95) !important;
        opacity: 0.9 !important;
    }

    /* Plan Mode toggle — overlaid INSIDE the chat input bar on the left.
       In Streamlit 1.58 st.toggle renders as data-testid="stCheckbox". */
    [data-testid="stMainBlockContainer"] [data-testid="stCheckbox"] {
        position: fixed !important;
        bottom: 4.5rem !important;
        left: 6.5rem !important;
        z-index: 1002 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stCheckbox"] p {
        font-size: 0.72rem;
        font-weight: 500;
        opacity: 0.55;
        margin: 0;
        white-space: nowrap;
    }
    /* Push textarea text right so it doesn't overlap with the toggle (~85px wide) */
    [data-testid="stChatInput"] textarea {
        padding-left: 6rem !important;
    }
    /* Bottom padding for main content area */
    .main .block-container {
        padding-bottom: 90px !important;
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
if "plan_mode" not in st.session_state:
    st.session_state.plan_mode = False
if "current_plan" not in st.session_state:
    st.session_state.current_plan = None
if "exec_cursor" not in st.session_state:
    st.session_state.exec_cursor = -1
if "exec_results" not in st.session_state:
    st.session_state.exec_results = []
if "exec_steps" not in st.session_state:
    st.session_state.exec_steps = []

def _clear_plan_widget_state():
    prefixes = ("plan_include_", "plan_input_", "plan_comment_", "plan_insert_open_", "plan_insert_desc_", "plan_fb_open_", "plan_fb_btn_", "plan_newmet_", "newmet_results_", "newmet_error_")
    for k in list(st.session_state.keys()):
        if any(k.startswith(p) for p in prefixes):
            del st.session_state[k]


def _collect_feedback(plan: dict) -> str:
    lines = []
    for i, step in enumerate(plan["plan"]):
        included = st.session_state.get(f"plan_include_{i}", step.get("included", True))
        comment  = st.session_state.get(f"plan_comment_{i}", "").strip()
        insert   = st.session_state.get(f"plan_insert_desc_{i}", "").strip()
        if not included:
            lines.append(f"Step {step['step_number']} ({step['step_name']}): Delete this step.")
        elif comment:
            lines.append(f"Step {step['step_number']} ({step['step_name']}): {comment}")
        if insert:
            lines.append(f"[Insert after step {step['step_number']}]: {insert}")
    return "\n".join(lines) if lines else "No changes requested."


def _insert_divider(i: int):
    """Render a compact '+' row between step cards."""
    _, mid, _ = st.columns([5, 1, 5])
    with mid:
        if st.button("＋", key=f"plan_insert_btn_{i}", help="Insert a step here", use_container_width=True):
            st.session_state[f"plan_insert_open_{i}"] = True
            st.rerun()
    if st.session_state.get(f"plan_insert_open_{i}"):
        _, inp, cls = st.columns([1, 7, 1])
        with inp:
            st.text_input("Describe the step to insert", key=f"plan_insert_desc_{i}", label_visibility="collapsed",
                          placeholder="Describe the step to insert here…")
        with cls:
            if st.button("✕", key=f"plan_insert_close_{i}", help="Cancel"):
                st.session_state[f"plan_insert_open_{i}"] = False
                st.rerun()


def plan_mode_ui():
    import copy

    plan         = st.session_state.current_plan
    exec_cursor  = st.session_state.exec_cursor
    exec_results = st.session_state.exec_results
    exec_steps   = st.session_state.exec_steps

    # ── Execution phase ───────────────────────────────────────────────────────
    if exec_cursor >= 0 and exec_steps:
        st.markdown("### Executing Plan")
        for k in range(exec_cursor):
            step = exec_steps[k]
            with st.container(border=True):
                if k < len(exec_results):
                    past_result = exec_results[k]
                    past_interp = past_result.get("__interpretation__", "")
                    past_display = {k2: v for k2, v in past_result.items() if k2 != "__interpretation__"}
                    if "error" in past_display:
                        st.error(f"Step {step['step_number']}: {step['step_name']} — failed")
                        st.error(past_display["error"])
                    else:
                        st.success(f"Step {step['step_number']}: {step['step_name']} — done")
                        if past_interp:
                            st.markdown(past_interp)
                        with st.expander("Raw output", expanded=False):
                            st.json(past_display, expanded=False)
                else:
                    st.success(f"Step {step['step_number']}: {step['step_name']} — done")

        if exec_cursor < len(exec_steps):
            step = exec_steps[exec_cursor]
            if len(exec_results) <= exec_cursor:
                with st.spinner(f"Running Step {step['step_number']}: {step['step_name']}…"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/plan/execute_step",
                            json={"step": step},
                            timeout=300,
                        )
                        if r.status_code == 200:
                            api_resp = r.json()
                            result = api_resp.get("result", api_resp)
                            interp = api_resp.get("interpretation", "")
                            if interp:
                                result = dict(result, __interpretation__=interp)
                        else:
                            result = {"error": r.text}
                    except Exception as e:
                        result = {"error": str(e)}
                exec_results = exec_results + [result]
                st.session_state.exec_results = exec_results
                st.rerun()
            else:
                result = exec_results[exec_cursor]
                interpretation = result.get("__interpretation__", "")
                display_result = {k: v for k, v in result.items() if k != "__interpretation__"}
                is_manual = display_result.get("status") == "manual"
                is_error = "error" in display_result and not is_manual

                retry_attempt = st.session_state.get(f"retry_attempt_{exec_cursor}", 0)
                retry_history = st.session_state.get(f"retry_history_{exec_cursor}", [])
                retry_stopped = st.session_state.get(f"retry_stopped_{exec_cursor}", False)
                MAX_RETRIES = 5
                can_retry = is_error and retry_attempt < MAX_RETRIES and not retry_stopped

                with st.container(border=True):
                    st.markdown(f"**Step {step['step_number']}: {step['step_name']}**")

                    if can_retry:
                        retry_label = (
                            "Step failed. Starting agent-assisted retry…"
                            if retry_attempt == 0 else
                            f"Attempt {retry_attempt} failed. Agent retrying ({retry_attempt + 1}/{MAX_RETRIES})…"
                        )
                        st.warning(retry_label)
                        with st.expander("Error details", expanded=False):
                            st.error(display_result.get("error", ""))
                        if st.button("Stop retries", key=f"stop_retry_{exec_cursor}_{retry_attempt}"):
                            st.session_state[f"retry_stopped_{exec_cursor}"] = True
                            st.rerun()
                        with st.spinner(f"Agent fix attempt {retry_attempt + 1}/{MAX_RETRIES}…"):
                            try:
                                rr = requests.post(
                                    f"{API_BASE}/plan/retry_step",
                                    json={
                                        "step": step,
                                        "error": display_result.get("error", ""),
                                        "history": retry_history,
                                    },
                                    timeout=300,
                                )
                                if rr.status_code == 200:
                                    rd = rr.json()
                                    new_result = rd.get("result", {"error": rr.text})
                                    fix_applied = rd.get("fix_applied", "")
                                    new_interp = rd.get("interpretation", "")
                                    if new_interp and "error" not in new_result:
                                        new_result = dict(new_result, __interpretation__=new_interp)
                                else:
                                    new_result = {"error": rr.text}
                                    fix_applied = "API error"
                            except Exception as _re:
                                new_result = {"error": str(_re)}
                                fix_applied = "exception"
                        new_history = retry_history + [{"fix_applied": fix_applied, "error": display_result.get("error", "")}]
                        st.session_state[f"retry_attempt_{exec_cursor}"] = retry_attempt + 1
                        st.session_state[f"retry_history_{exec_cursor}"] = new_history
                        updated = list(exec_results)
                        updated[exec_cursor] = new_result
                        st.session_state.exec_results = updated
                        st.rerun()

                    elif is_error:
                        if retry_attempt >= MAX_RETRIES:
                            st.error(f"All {MAX_RETRIES} retry attempts exhausted.")
                        elif retry_stopped:
                            st.info("Retries stopped by user.")
                        st.error(display_result.get("error", ""))
                        if retry_history:
                            with st.expander(f"Retry history ({len(retry_history)} attempt(s))", expanded=False):
                                for _i, _h in enumerate(retry_history, 1):
                                    st.markdown(f"**Attempt {_i}:** {_h.get('fix_applied', 'N/A')}")
                                    st.caption(_h.get("error", ""))

                    elif is_manual:
                        st.info(display_result.get("message", "This step requires manual execution outside the app."))
                        st.divider()
                        st.markdown("**To perform this step with the current model:**")
                        dl_col, ul_col = st.columns(2)

                        with dl_col:
                            st.markdown("**1 — Download current model**")
                            try:
                                dl_resp = requests.get(f"{API_BASE}/download_model/", timeout=30)
                                if dl_resp.status_code == 200:
                                    cd = dl_resp.headers.get("content-disposition", "")
                                    dl_name = cd.split("filename=")[-1].strip('"') if "filename=" in cd else "model.xml"
                                    st.download_button(
                                        "Download model (.xml)",
                                        data=dl_resp.content,
                                        file_name=dl_name,
                                        mime="application/xml",
                                        use_container_width=True,
                                    )
                                else:
                                    st.warning("No model loaded yet to download.")
                            except Exception as _e:
                                st.warning(f"Download unavailable: {_e}")

                        with ul_col:
                            st.markdown("**2 — Upload modified model**")
                            upload_done_key = f"manual_upload_done_{exec_cursor}"
                            if st.session_state.get(upload_done_key):
                                st.success("Modified model uploaded.")
                            else:
                                uploaded = st.file_uploader(
                                    "Upload modified model",
                                    type=["xml", "sbml"],
                                    key=f"manual_upload_{exec_cursor}",
                                    label_visibility="collapsed",
                                )
                                if uploaded is not None:
                                    if st.button("Confirm upload", key=f"manual_upload_btn_{exec_cursor}", use_container_width=True):
                                        ul_resp = requests.post(
                                            f"{API_BASE}/upload_model/",
                                            files={"file": (uploaded.name, uploaded.getvalue(), "application/xml")},
                                            timeout=60,
                                        )
                                        if ul_resp.status_code == 200:
                                            st.session_state[upload_done_key] = True
                                            st.rerun()
                                        else:
                                            st.error(f"Upload failed: {ul_resp.text}")

                    else:
                        if retry_attempt > 0:
                            st.success(f"Fixed after {retry_attempt} retry attempt(s).")
                        if interpretation:
                            st.markdown(interpretation)
                        with st.expander("Raw output", expanded=not interpretation):
                            st.json(display_result, expanded=True)

                c1, c2 = st.columns(2)
                with c1:
                    is_last = exec_cursor == len(exec_steps) - 1
                    btn_label = "Finish" if is_last else "Continue to next step"
                    if st.button(btn_label, use_container_width=True, type="primary"):
                        st.session_state.exec_cursor = exec_cursor + 1
                        st.rerun()
                with c2:
                    if st.button("Stop execution", use_container_width=True):
                        st.session_state.exec_cursor = len(exec_steps)
                        st.rerun()
        else:
            st.success("All steps executed.")
            if st.button("Back to plan editor"):
                st.session_state.exec_cursor = -1
                st.rerun()
        return

    if plan is None:
        return

    # ── Phase B: Interactive plan editor ─────────────────────────────────────
    st.divider()
    st.caption(plan.get("summary", ""))

    steps = plan["plan"]

    # "+" before the very first step
    _insert_divider(-1)

    for i, step in enumerate(steps):
        # Initialise widget state once (won't overwrite user edits on rerun)
        if f"plan_include_{i}" not in st.session_state:
            st.session_state[f"plan_include_{i}"] = step.get("included", True)
        for param, spec in step.get("user_inputs", {}).items():
            key = f"plan_input_{i}_{param}"
            if key not in st.session_state:
                st.session_state[key] = str(spec["value"]) if spec.get("value") is not None else ""

        included = st.session_state[f"plan_include_{i}"]

        # ── Excluded: compact pill with restore ──────────────────────────────
        if not included:
            with st.container():
                c_lbl, c_rst = st.columns([9, 1])
                with c_lbl:
                    st.markdown(
                        f"<span style='color:grey;text-decoration:line-through'>"
                        f"Step {step['step_number']}: {step['step_name']}</span> — *excluded*",
                        unsafe_allow_html=True,
                    )
                with c_rst:
                    if st.button("↩", key=f"plan_restore_{i}", help="Restore step"):
                        st.session_state[f"plan_include_{i}"] = True
                        st.rerun()
            _insert_divider(i)
            continue

        # ── Active step card ─────────────────────────────────────────────────
        with st.container(border=True):
            # Header row: step name | feedback toggle | delete
            h_name, h_fb, h_del = st.columns([10, 1, 1])
            with h_name:
                st.markdown(f"**Step {step['step_number']}: {step['step_name']}**")
            with h_fb:
                fb_open_key = f"plan_fb_open_{i}"
                if fb_open_key not in st.session_state:
                    st.session_state[fb_open_key] = False
                if st.button("💬", key=f"plan_fb_btn_{i}", help="Add feedback for re-planning", use_container_width=True):
                    st.session_state[fb_open_key] = not st.session_state[fb_open_key]
                    st.rerun()
            with h_del:
                if st.button("✕", key=f"plan_delete_{i}", help="Remove this step", use_container_width=True):
                    st.session_state[f"plan_include_{i}"] = False
                    st.rerun()

            # Inline feedback area — rendered directly so session_state is always reliable
            if st.session_state.get(fb_open_key):
                st.text_area(
                    "Feedback for re-planning",
                    key=f"plan_comment_{i}",
                    height=80,
                    placeholder="Describe the change you want for this step…",
                    label_visibility="collapsed",
                )

            # Two-column body
            col_desc, col_tool = st.columns([1, 1])

            with col_desc:
                st.markdown("**Description**")
                st.markdown(step.get("description", "—"))
                if step.get("rationale"):
                    st.caption(f"*{step['rationale']}*")

            with col_tool:
                st.markdown("**Tool & Source**")
                st.code(step["tool"], language=None)
                st.caption(
                    f"Source: {step.get('source_paper','—')} / {step.get('source_step_id','—')}"
                )
                if step.get("user_inputs"):
                    st.markdown("**Inputs**")
                    for param, spec in step["user_inputs"].items():
                        req = " *" if spec.get("required") else ""
                        label = param.replace("_", " ").title() + req
                        st.text_input(
                            label,
                            help=spec.get("description"),
                            key=f"plan_input_{i}_{param}",
                        )

                # Dynamic new-metabolite fields for add_reaction steps
                if step.get("tool") == "add_reaction":
                    results_key = f"newmet_results_{i}"
                    error_key = f"newmet_error_{i}"

                    if st.button("Check metabolites", key=f"detect_newmet_{i}", help="Detect which metabolites in the equation are not yet in the model"):
                        import copy as _copy
                        equation_val = st.session_state.get(f"plan_input_{i}_equation", "").strip()
                        if not equation_val:
                            st.session_state[error_key] = "Enter an equation first."
                            st.session_state[results_key] = None
                        else:
                            try:
                                r = requests.post(f"{API_BASE}/tools/detect_new_metabolites", json={"equation": equation_val}, timeout=10)
                                data = r.json()

                                if data.get("error") == "no_model":
                                    # Find a load_model step earlier in the plan and auto-run it
                                    load_step = None
                                    load_idx = None
                                    for j, prev in enumerate(steps):
                                        if j >= i:
                                            break
                                        if prev.get("tool") == "load_model":
                                            load_step = _copy.deepcopy(prev)
                                            load_idx = j
                                            break

                                    if load_step is not None:
                                        session_mid = st.session_state.get(f"plan_input_{load_idx}_model_id", "").strip()
                                        if session_mid:
                                            load_step["user_inputs"]["model_id"]["value"] = session_mid
                                        model_id_val = (load_step.get("user_inputs", {}).get("model_id") or {}).get("value")

                                        if model_id_val:
                                            load_r = requests.post(f"{API_BASE}/plan/execute_step", json={"step": load_step}, timeout=60)
                                            if load_r.status_code == 200:
                                                r2 = requests.post(f"{API_BASE}/tools/detect_new_metabolites", json={"equation": equation_val}, timeout=10)
                                                data = r2.json()
                                            else:
                                                data = {"error": "load_failed", "message": f"Failed to load model: {load_r.json().get('detail', load_r.text)}"}
                                        else:
                                            data = {"error": "no_model", "message": "Fill in the Model Id field in the load model step above first."}

                                if data.get("error"):
                                    st.session_state[error_key] = data.get("message", "Could not check metabolites.")
                                    st.session_state[results_key] = None
                                else:
                                    st.session_state[error_key] = None
                                    st.session_state[results_key] = data.get("new_metabolites", [])
                            except Exception as ex:
                                st.session_state[error_key] = f"Could not reach backend: {ex}"
                                st.session_state[results_key] = None
                        st.rerun()

                    err = st.session_state.get(error_key)
                    if err:
                        st.warning(err)
                    new_mets = st.session_state.get(results_key)
                    if new_mets is not None:
                        if new_mets:
                            st.markdown("**New metabolites — provide details:**")
                            for met_id in new_mets:
                                st.text_input(
                                    met_id,
                                    help="formula | name | compartment — e.g. C25H39N2O10PRS|Malonyl-ACP|c",
                                    key=f"plan_newmet_{i}_{met_id}",
                                    placeholder="formula|name|compartment",
                                )
                        else:
                            st.success("All metabolites already exist in the model.")

        # "+" between steps
        _insert_divider(i)

    st.divider()
    col_replan, col_execute = st.columns(2)

    with col_replan:
        if st.button("Re-plan with feedback", use_container_width=True):
            feedback = _collect_feedback(plan)
            with st.spinner("Re-planning…"):
                try:
                    r = requests.post(
                        f"{API_BASE}/plan/revise",
                        json={"query": plan["query"], "current_plan": plan, "feedback": feedback},
                        timeout=300,
                    )
                    if r.status_code == 200:
                        _clear_plan_widget_state()
                        st.session_state.current_plan = r.json()["plan"]
                        st.rerun()
                    else:
                        st.error(f"Revision failed: {r.json().get('detail', r.text)}")
                except Exception as e:
                    st.error(f"Backend error: {e}")

    with col_execute:
        if st.button("Execute Plan", use_container_width=True, type="primary"):
            steps_to_run = []
            for i, step in enumerate(plan["plan"]):
                if not st.session_state.get(f"plan_include_{i}", step.get("included", True)):
                    continue
                s = copy.deepcopy(step)
                for param in s.get("user_inputs", {}):
                    val = st.session_state.get(f"plan_input_{i}_{param}", "").strip()
                    if val:
                        s["user_inputs"][param]["value"] = val
                # Collect dynamic new_metabolites_info for add_reaction steps
                if s.get("tool") == "add_reaction":
                    new_mets = st.session_state.get(f"newmet_results_{i}") or []
                    entries = [
                        f"{mid}|{st.session_state.get(f'plan_newmet_{i}_{mid}', '').strip()}"
                        for mid in new_mets
                        if st.session_state.get(f"plan_newmet_{i}_{mid}", "").strip()
                    ]
                    if entries:
                        s.setdefault("user_inputs", {})["new_metabolites_info"] = {
                            "value": ", ".join(entries),
                            "required": False,
                            "description": "auto-collected from metabolite detail fields",
                        }
                steps_to_run.append(s)
            st.session_state.exec_steps   = steps_to_run
            st.session_state.exec_cursor  = 0
            st.session_state.exec_results = []
            st.rerun()


@st.dialog("🗂️ Name Your Session")
def name_session_dialog():
    st.markdown("Give this session a descriptive name. All artifacts (FVA, knockouts, sampling) will be saved in a folder with this name under `artifacts/`.")
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
    
    provider = st.selectbox("Choose LLM Provider", ["groq", "ollama", "openai", "Hugging Face", "gemini", "llama.cpp"])
    model_label = "Model Path (.gguf file)" if provider == "llama.cpp" else "Model Name"
    model_placeholder = "/path/to/model.gguf" if provider == "llama.cpp" else ""
    model = st.text_input(
        model_label,
        value="llama-3.1-8b-instant" if provider == "groq" else "gemini-3.1-pro-preview" if provider == "gemini" else "",
        placeholder=model_placeholder,
    )
    api_key = None

    if provider == "openai":
        api_key = st.text_input("Enter API Key", type="password")
    elif provider == "Hugging Face":
        api_key = st.text_input("Enter HF TOKEN", type="password")
    elif provider == "groq":
        api_key = st.text_input("Enter API Key", type="password")
    elif provider == "gemini":
        api_key = st.text_input("Enter Gemini API Key", type="password")

    if st.button("Apply LLM"):
        if provider in ['groq', 'openai', 'Hugging Face', 'gemini'] and not api_key:
            st.error("API Key is required for Groq, OpenAI, Hugging Face, and Gemini providers.")
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
                    st.success(f"Artifacts saved in `artifacts/{st.session_state.session_name}`.")
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
    # ── Header: title + 3 toolbar icon buttons ───────────────────────────────
    col1, col2, col3, col4 = st.columns([84, 4, 4, 4])
    with col1:
        header_title = "📋 Plan Mode" if st.session_state.plan_mode else "💬 Agentic Chat"
        st.header(header_title)
    with col2:
        st.markdown('<div class="toolbar-btn">', unsafe_allow_html=True)
        if st.button("📁", help="Model Management", use_container_width=True, key="btn_models"):
            model_management_dialog()
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="toolbar-btn">', unsafe_allow_html=True)
        if st.button("🧠", help="LLM Configuration", use_container_width=True, key="btn_llm"):
            llm_configuration_dialog()
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="toolbar-btn">', unsafe_allow_html=True)
        if st.button("⏹", help="End Session", use_container_width=True, key="btn_end"):
            end_session_dialog()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.llm:
        sub = f"Provider: {st.session_state.get('llm_provider','?')} | Model: {st.session_state.get('llm_model','?')} | Session: {st.session_state.get('session_name','?')}"
        st.caption(sub)

        # ── Main content area (chat or plan) ──────────────────────────────────
        if st.session_state.plan_mode:
            plan_mode_ui()
        else:
            for role, msg in st.session_state.chat_history:
                if role == "user":
                    st.chat_message("user").write(msg)
                else:
                    st.chat_message("assistant").write(msg)

        # ── Plan Mode toggle (CSS-fixed to bottom-right, above chat input) ────
        _pm = st.session_state.plan_mode
        new_pm = st.toggle(
            "Plan",
            value=_pm,
            key="plan_mode_toggle",
            help="Generate a step-by-step analysis plan instead of chatting directly",
        )
        if new_pm != _pm:
            st.session_state.plan_mode = new_pm
            if not new_pm:
                st.session_state.current_plan = None
                st.session_state.exec_cursor = -1
                st.session_state.exec_results = []
                st.session_state.exec_steps = []
                _clear_plan_widget_state()
            st.rerun()

        placeholder = "What do you want to achieve?" if _pm else "Ask about the model…"
        user_input = st.chat_input(placeholder)

        if user_input:
            if _pm:
                with st.spinner("Building plan…"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/plan/generate",
                            json={"query": user_input.strip()},
                            timeout=300,
                        )
                        if r.status_code == 200:
                            _clear_plan_widget_state()
                            st.session_state.current_plan = r.json()["plan"]
                            st.session_state.exec_cursor = -1
                            st.session_state.exec_results = []
                            st.session_state.exec_steps = []
                        else:
                            st.error(f"Plan generation failed: {r.json().get('detail', r.text)}")
                    except Exception as e:
                        st.error(f"Backend error: {e}")
            else:
                msg = user_input.strip()
                st.session_state.chat_history.append(("user", msg))
                with st.spinner("Agent is thinking…"):
                    try:
                        res = requests.post(f"{API_BASE}/chat/", json={"message": msg})
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
            st.rerun()
    else:
        st.warning("Please configure the LLM in the sidebar to start chatting.")
        st.info("Supported providers: groq, ollama, openai, Hugging Face, gemini, llama.cpp")