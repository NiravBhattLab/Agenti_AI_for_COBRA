import streamlit as st
import requests
import os

API_BASE = os.environ.get("API_BASE", "http://localhost:8000")

# Plan steps can run genome-scale reconstructions (e.g. CORDA on Recon3D /
# Human-GEM) that legitimately take many minutes. Use a generous client timeout
# so the frontend doesn't abandon a step that the backend is still computing.
PLAN_EXEC_TIMEOUT = 3600  # seconds

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
    /* Push textarea text right so it doesn't overlap with Plan toggle + attach button */
    [data-testid="stChatInput"] textarea {
        padding-left: 9rem !important;
    }
    /* Bottom padding for main content area */
    .main .block-container {
        padding-bottom: 90px !important;
    }

    /* File attach '+' button — same fixed-overlay trick as Plan toggle.
       st.markdown('<span class="chat-attach-marker">') renders inside an element-container
       as a sibling to the button's element-container. :has() lets us grab that next sibling
       and apply position:fixed to it (the container, not the button child). */
    [data-testid="element-container"]:has(.chat-attach-marker) + [data-testid="element-container"] {
        position: fixed !important;
        left: 11rem !important;
        bottom: 4.5rem !important;
        z-index: 1002 !important;
        width: auto !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    [data-testid="element-container"]:has(.chat-attach-marker) + [data-testid="element-container"] button {
        border-radius: 50% !important;
        width: 2.2rem !important;
        height: 2.2rem !important;
        min-width: 0 !important;
        padding: 0 !important;
        font-size: 1.2rem !important;
        line-height: 1 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border: none !important;
        background: transparent !important;
        color: inherit !important;
        opacity: 0.65;
        transition: opacity 0.15s, background 0.15s, transform 0.1s !important;
        cursor: pointer !important;
    }
    [data-testid="element-container"]:has(.chat-attach-marker) + [data-testid="element-container"] button:hover {
        background: rgba(255,255,255,0.1) !important;
        opacity: 1 !important;
        transform: scale(1.12) !important;
    }
    [data-testid="element-container"]:has(.chat-attach-marker) + [data-testid="element-container"] button:active {
        transform: scale(0.95) !important;
        opacity: 0.9 !important;
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
    st.session_state.disclaimer = True
if "app_launched" not in st.session_state:
    st.session_state.app_launched = False
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
if "pending_chat_upload" not in st.session_state:
    st.session_state.pending_chat_upload = None  # None or {"param", "file_types", "description"}
if "chat_upload_sig" not in st.session_state:
    st.session_state.chat_upload_sig = None  # "{name}:{size}" sentinel to prevent duplicate uploads on rerun
if "chat_attached_file" not in st.session_state:
    st.session_state.chat_attached_file = None  # filename staged for next message
if "chat_attach_sig" not in st.session_state:
    st.session_state.chat_attach_sig = None
if "pending_tool_inputs" not in st.session_state:
    st.session_state.pending_tool_inputs = None  # None or {"tool_name", "explanation", "param_specs"}
if "tool_inputs_original_msg" not in st.session_state:
    st.session_state.tool_inputs_original_msg = None

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
                            timeout=PLAN_EXEC_TIMEOUT,
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
                                    timeout=PLAN_EXEC_TIMEOUT,
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
            st.markdown("### Plan Complete")
            st.markdown(f"> **Your question:** {plan['query']}")
            st.divider()

            if "plan_final_summary" not in st.session_state:
                with st.spinner("Generating answer…"):
                    try:
                        r = requests.post(
                            f"{API_BASE}/plan/summarize",
                            json={
                                "query": plan["query"],
                                "plan_summary": plan.get("summary", ""),
                                "steps": exec_steps,
                                "results": exec_results,
                            },
                            timeout=60,
                        )
                        st.session_state.plan_final_summary = (
                            r.json().get("summary", "") if r.status_code == 200 else ""
                        )
                    except Exception:
                        st.session_state.plan_final_summary = ""

            if st.session_state.get("plan_final_summary"):
                st.markdown(st.session_state.plan_final_summary)
            else:
                st.success("All steps executed.")

            with st.expander("Step-by-step results", expanded=False):
                for i, step in enumerate(exec_steps):
                    res = exec_results[i] if i < len(exec_results) else {}
                    interp = res.get("__interpretation__", "")
                    if "error" in res:
                        st.error(f"**{step['step_name']}** — failed: {res['error']}")
                    else:
                        st.success(f"**{step['step_name']}**")
                        if interp:
                            st.markdown(interp)

            st.divider()
            if st.button("Back to plan editor"):
                st.session_state.exec_cursor = -1
                st.session_state.pop("plan_final_summary", None)
                st.rerun()
        return

    if plan is None:
        return

    # ── Phase B: Interactive plan editor ─────────────────────────────────────
    if plan.get("query"):
        st.markdown(f"> {plan['query']}")
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
                    _params = list(step["user_inputs"].items())
                    _j = 0
                    while _j < len(_params):
                        _param, _spec = _params[_j]
                        _req = " *" if _spec.get("required") else ""
                        _label = _param.replace("_", " ").title() + _req

                        if _spec.get("input_type") == "file":
                            # File inputs stay full-width
                            text_key = f"plan_input_{i}_{_param}"
                            up = st.file_uploader(
                                _label,
                                type=_spec.get("file_types"),
                                help=_spec.get("description"),
                                key=f"plan_file_{i}_{_param}",
                            )
                            if up is not None:
                                sig_key = f"plan_file_sig_{i}_{_param}"
                                sig = f"{up.name}:{up.size}"
                                if st.session_state.get(sig_key) != sig:
                                    try:
                                        res = requests.post(
                                            f"{API_BASE}/upload_data_file/",
                                            files={"file": (up.name, up.getvalue())},
                                            timeout=120,
                                        )
                                        if res.status_code == 200:
                                            st.session_state[text_key] = res.json()["filename"]
                                            st.session_state[sig_key] = sig
                                        else:
                                            st.error(f"Upload failed: {res.json().get('detail', res.text)}")
                                    except Exception as ex:
                                        st.error(f"Upload failed: {ex}")
                            current = st.session_state.get(text_key, "")
                            if current:
                                st.caption(f"Uploaded: `{current}`")
                            _j += 1
                        else:
                            # Pair consecutive text inputs into two columns
                            _next_is_text = (
                                _j + 1 < len(_params) and
                                _params[_j + 1][1].get("input_type") != "file"
                            )
                            if _next_is_text:
                                _np, _ns = _params[_j + 1]
                                _nlabel = _np.replace("_", " ").title() + (" *" if _ns.get("required") else "")
                                _ic1, _ic2 = st.columns(2)
                                with _ic1:
                                    _dv = _spec.get("value")
                                    _ph = (f"defaults to {_dv}" if _dv else "optional") if not _spec.get("required") else None
                                    st.text_input(_label, help=_spec.get("description"), key=f"plan_input_{i}_{_param}", placeholder=_ph)
                                with _ic2:
                                    _ndv = _ns.get("value")
                                    _nph = (f"defaults to {_ndv}" if _ndv else "optional") if not _ns.get("required") else None
                                    st.text_input(_nlabel, help=_ns.get("description"), key=f"plan_input_{i}_{_np}", placeholder=_nph)
                                _j += 2
                            else:
                                _dv = _spec.get("value")
                                _ph = (f"optional, defaults to {_dv}" if _dv else "optional") if not _spec.get("required") else None
                                st.text_input(_label, help=_spec.get("description"), key=f"plan_input_{i}_{_param}", placeholder=_ph)
                                _j += 1

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

def welcome_page():
    st.markdown(
        """
        <style>
        /* ── Welcome page global resets ── */
        .mp-page {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 3vh 1rem 2vh 1rem;
            text-align: center;
        }
        /* Title */
        .mp-title {
            font-size: 4rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            line-height: 1;
            margin: 0 0 0.75rem 0;
        }
        .mp-title-meta { color: #e8e8e8; }
        .mp-title-pilot {
            background: linear-gradient(135deg, #ff6b6b 0%, #ffa94d 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        /* Tagline */
        .mp-tagline {
            font-size: 0.95rem;
            color: #686878;
            line-height: 1.55;
            max-width: 480px;
            margin: 0 auto 1.6rem auto;
        }
        /* Mode cards row */
        .mp-cards {
            display: flex;
            gap: 1rem;
            justify-content: center;
            width: 100%;
            max-width: 660px;
            margin: 0 auto 1.4rem auto;
        }
        .mp-card {
            flex: 1;
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 1.1rem 1.2rem 1rem 1.2rem;
            text-align: left;
            position: relative;
            overflow: hidden;
        }
        .mp-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: linear-gradient(90deg, #ff6b6b, #ffa94d);
            opacity: 0;
            transition: opacity 0.25s;
        }
        .mp-card:hover::before { opacity: 1; }
        .mp-card:hover { border-color: rgba(255, 107, 107, 0.25); }
        .mp-card-icon {
            font-size: 1.5rem;
            margin-bottom: 0.45rem;
            display: block;
        }
        .mp-card-name {
            font-size: 0.95rem;
            font-weight: 700;
            color: #ddd;
            margin: 0 0 0.55rem 0;
            letter-spacing: 0.01em;
        }
        .mp-card-desc {
            font-size: 0.83rem;
            color: #666;
            line-height: 1.65;
            margin: 0;
        }
        /* Style the Streamlit primary button on this page */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: linear-gradient(135deg, #ff6b6b 0%, #ff8c42 100%);
            border: none;
            border-radius: 10px;
            font-size: 0.95rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            padding: 0.65rem 2.5rem;
            box-shadow: 0 4px 20px rgba(255, 107, 107, 0.3);
            transition: box-shadow 0.2s, transform 0.15s;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            box-shadow: 0 6px 28px rgba(255, 107, 107, 0.45);
            transform: translateY(-1px);
        }
        </style>

        <div class="mp-page">
            <h1 class="mp-title">
                <span class="mp-title-meta">Meta</span><span class="mp-title-pilot">Pilot</span>
            </h1>
            <p class="mp-tagline">
                An AI agent for constraint-based metabolic modelling. Analyse COBRA-compatible
                models through natural language, or define a research objective and let the
                agent plan and execute the workflow automatically.
            </p>
            <div class="mp-cards">
                <div class="mp-card">
                    <span class="mp-card-icon">💬</span>
                    <p class="mp-card-name">MetaInteract</p>
                    <p class="mp-card-desc">
                        Direct conversational interface — chat with the AI agent to query,
                        manipulate, and analyse metabolic models in real time, one message
                        at a time.
                    </p>
                </div>
                <div class="mp-card">
                    <span class="mp-card-icon">📋</span>
                    <p class="mp-card-name">MetaPlan</p>
                    <p class="mp-card-desc">
                        Planning mode — describe a high-level analysis goal and the agent
                        generates, then executes, a structured step-by-step plan
                        automatically.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Launch button — centred via columns
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        if st.button("Launch  →", type="primary", use_container_width=True):
            st.session_state.app_launched = True
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

@st.dialog("📎 Attach a data file")
def attach_file_dialog():
    st.caption("The filename will be included in your next message so the agent can use it.")
    f = st.file_uploader(
        "Upload file",
        type=["csv", "tsv", "txt", "xml", "sbml", "faa", "fasta", "fa", "fna"],
        key="dialog_attach_file",
        label_visibility="collapsed",
    )
    if f is not None:
        if st.button("Attach", type="primary", use_container_width=True, key="dialog_attach_btn"):
            sig = f"{f.name}:{f.size}"
            if st.session_state.chat_attach_sig != sig:
                try:
                    res = requests.post(
                        f"{API_BASE}/upload_data_file/",
                        files={"file": (f.name, f.getvalue())},
                        timeout=120,
                    )
                    if res.status_code == 200:
                        st.session_state.chat_attached_file = res.json()["filename"]
                        st.session_state.chat_attach_sig = sig
                        st.rerun()
                    else:
                        st.error(f"Upload failed: {res.json().get('detail', res.text)}")
                except Exception as ex:
                    st.error(f"Upload failed: {ex}")


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


if not st.session_state.app_launched:
    welcome_page()
elif not st.session_state.session_ready:
    name_session_dialog()
else:
    # ── Header: title + 3 toolbar icon buttons ───────────────────────────────
    col1, col2, col3, col4 = st.columns([84, 4, 4, 4])
    with col1:
        header_title = "📋 MetaPlan" if st.session_state.plan_mode else "💬 MetaInteract"
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

            # ── Attached file badge (shown when a file is staged via the + button) ──
            if st.session_state.chat_attached_file:
                c_badge, c_rm = st.columns([12, 1])
                with c_badge:
                    st.caption(f"📎 `{st.session_state.chat_attached_file}` — will be sent with your next message")
                with c_rm:
                    if st.button("✕", key="chat_attach_clear", help="Remove attached file"):
                        st.session_state.chat_attached_file = None
                        st.session_state.chat_attach_sig = None
                        st.rerun()

            # ── Inline file upload widget for chat mode ────────────────────────
            if st.session_state.pending_chat_upload is not None:
                pup = st.session_state.pending_chat_upload
                with st.container(border=True):
                    st.markdown(f"**Upload required:** {pup['description']}")
                    file_types = [ft.strip().lstrip(".") for ft in pup["file_types"]] if pup.get("file_types") else None
                    uploaded_file = st.file_uploader(
                        f"File for `{pup['param']}`",
                        type=file_types,
                        key="chat_upload_widget",
                    )
                    c_confirm, c_cancel = st.columns(2)
                    with c_confirm:
                        if uploaded_file is not None:
                            sig = f"{uploaded_file.name}:{uploaded_file.size}"
                            if st.button("Upload & Continue", key="chat_upload_confirm", type="primary", use_container_width=True):
                                if st.session_state.chat_upload_sig != sig:
                                    try:
                                        res = requests.post(
                                            f"{API_BASE}/upload_data_file/",
                                            files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                                            timeout=120,
                                        )
                                        if res.status_code == 200:
                                            filename = res.json()["filename"]
                                            st.session_state.chat_upload_sig = sig
                                            st.session_state.pending_chat_upload = None
                                            follow_up = (
                                                f"The file has been uploaded. Filename: {filename}. "
                                                f"Please proceed using this file for the {pup['param']} parameter."
                                            )
                                            st.session_state.chat_history.append(("user", f"[Uploaded: {filename}]"))
                                            with st.spinner("Agent is processing uploaded file…"):
                                                try:
                                                    agent_res = requests.post(f"{API_BASE}/chat/", json={"message": follow_up}, timeout=300)
                                                    if agent_res.status_code == 200:
                                                        data = agent_res.json()
                                                        if data.get("model_id"):
                                                            st.session_state.model_id = data["model_id"]
                                                        if data.get("needs_upload"):
                                                            st.session_state.pending_chat_upload = data["needs_upload"]
                                                            st.session_state.chat_upload_sig = None
                                                        st.session_state.chat_history.append(("agent", data["response"]))
                                                    else:
                                                        st.session_state.chat_history.append(("agent", f"⚠️ Error: {agent_res.text}"))
                                                except Exception as ex:
                                                    st.session_state.chat_history.append(("agent", f"⚠️ Exception: {ex}"))
                                            st.rerun()
                                        else:
                                            st.error(f"Upload failed: {res.json().get('detail', res.text)}")
                                    except Exception as ex:
                                        st.error(f"Upload failed: {ex}")
                    with c_cancel:
                        if st.button("Cancel", key="chat_upload_cancel", use_container_width=True):
                            st.session_state.pending_chat_upload = None
                            st.session_state.chat_upload_sig = None
                            st.session_state.chat_history.append(("agent", "Upload cancelled. You can ask again or provide a filename manually."))
                            st.rerun()

        # ── Inline tool inputs form for chat mode ─────────────────────────────
        if st.session_state.pending_tool_inputs is not None:
            pti = st.session_state.pending_tool_inputs
            tool_name = pti["tool_name"]
            param_specs = pti.get("param_specs", {})
            with st.container(border=True):
                st.markdown(f"**Provide details to run `{tool_name}`**")
                if pti.get("explanation"):
                    st.caption(pti["explanation"])
                _params = list(param_specs.items())
                _j = 0
                while _j < len(_params):
                    _param, _spec = _params[_j]
                    _req = " *" if _spec.get("required") else ""
                    _label = _param.replace("_", " ").title() + _req
                    _raw = _spec.get("value")
                    _val = "" if _raw is None else str(_raw)
                    _ph = ("optional" if _raw is None else f"defaults to {_raw}") if not _spec.get("required") else None
                    if _spec.get("input_type") == "file":
                        text_key = f"chat_tool_input_{_param}"
                        up = st.file_uploader(
                            _label,
                            type=_spec.get("file_types"),
                            help=_spec.get("description"),
                            key=f"chat_tool_file_{_param}",
                        )
                        if up is not None:
                            sig = f"{up.name}:{up.size}"
                            sig_key = f"chat_tool_file_sig_{_param}"
                            if st.session_state.get(sig_key) != sig:
                                try:
                                    res = requests.post(
                                        f"{API_BASE}/upload_data_file/",
                                        files={"file": (up.name, up.getvalue())},
                                        timeout=120,
                                    )
                                    if res.status_code == 200:
                                        st.session_state[text_key] = res.json()["filename"]
                                        st.session_state[sig_key] = sig
                                    else:
                                        st.error(f"Upload failed: {res.json().get('detail', res.text)}")
                                except Exception as ex:
                                    st.error(f"Upload failed: {ex}")
                        current = st.session_state.get(text_key, "")
                        if current:
                            st.caption(f"Uploaded: `{current}`")
                        _j += 1
                    else:
                        _next_is_text = (
                            _j + 1 < len(_params) and
                            _params[_j + 1][1].get("input_type") != "file"
                        )
                        if _next_is_text:
                            _np, _ns = _params[_j + 1]
                            _nlabel = _np.replace("_", " ").title() + (" *" if _ns.get("required") else "")
                            _nraw = _ns.get("value")
                            _nval = "" if _nraw is None else str(_nraw)
                            _nph = ("optional" if _nraw is None else f"defaults to {_nraw}") if not _ns.get("required") else None
                            _ic1, _ic2 = st.columns(2)
                            with _ic1:
                                st.text_input(_label, value=_val, help=_spec.get("description"), key=f"chat_tool_input_{_param}", placeholder=_ph)
                            with _ic2:
                                st.text_input(_nlabel, value=_nval, help=_ns.get("description"), key=f"chat_tool_input_{_np}", placeholder=_nph)
                            _j += 2
                        else:
                            st.text_input(_label, value=_val, help=_spec.get("description"), key=f"chat_tool_input_{_param}", placeholder=_ph)
                            _j += 1

                c_run, c_cancel = st.columns(2)
                with c_run:
                    if st.button("Run Tool", key="chat_tool_submit", type="primary", use_container_width=True):
                        params = {}
                        missing_required = []
                        for _param, _spec in param_specs.items():
                            if _spec.get("input_type") == "file":
                                val = st.session_state.get(f"chat_tool_input_{_param}", "").strip()
                            else:
                                val = st.session_state.get(f"chat_tool_input_{_param}", "").strip()
                            if val:
                                params[_param] = val
                            elif _spec.get("required"):
                                missing_required.append(_param)
                        if missing_required:
                            st.warning(f"Required fields missing: {', '.join(missing_required)}")
                        else:
                            with st.spinner(f"Running {tool_name}…"):
                                try:
                                    res = requests.post(
                                        f"{API_BASE}/chat/submit_tool_inputs",
                                        json={
                                            "tool_name": tool_name,
                                            "original_message": st.session_state.tool_inputs_original_msg or "",
                                            "params": params,
                                        },
                                        timeout=300,
                                    )
                                    if res.status_code == 200:
                                        data = res.json()
                                        if data.get("model_id"):
                                            st.session_state.model_id = data["model_id"]
                                        st.session_state.chat_history.append(("agent", data["response"]))
                                    else:
                                        st.session_state.chat_history.append(("agent", f"⚠️ Error: {res.json().get('detail', res.text)}"))
                                except Exception as ex:
                                    st.session_state.chat_history.append(("agent", f"⚠️ Exception: {ex}"))
                            st.session_state.pending_tool_inputs = None
                            st.session_state.tool_inputs_original_msg = None
                            st.rerun()
                with c_cancel:
                    if st.button("Cancel", key="chat_tool_cancel", use_container_width=True):
                        st.session_state.pending_tool_inputs = None
                        st.session_state.tool_inputs_original_msg = None
                        st.session_state.chat_history.append(("agent", "Operation cancelled. You can re-phrase your request with the specific details."))
                        st.rerun()

        # ── Plan Mode toggle (CSS-fixed to bottom-right, above chat input) ────
        _pm = st.session_state.plan_mode
        new_pm = st.toggle(
            "MetaPlan",
            value=_pm,
            key="plan_mode_toggle",
            help="Switch to MetaPlan mode to generate a step-by-step analysis plan instead of chatting directly",
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

        # ── File attach '+' button — CSS-fixed inside chat input bar ─────────
        # Marker span is a sibling element-container to the button.
        # CSS :has(.chat-attach-marker) + element-container applies position:fixed to the
        # button's wrapper — same mechanism as the Plan toggle (not the child button element).
        if not _pm:
            st.markdown('<span class="chat-attach-marker"></span>', unsafe_allow_html=True)
            if st.button("＋", key="chat_attach_open", help="Attach a data file"):
                attach_file_dialog()

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
                # Clear any pending upload or tool inputs form if the user types a new message
                if st.session_state.pending_chat_upload is not None:
                    st.session_state.pending_chat_upload = None
                    st.session_state.chat_upload_sig = None
                if st.session_state.pending_tool_inputs is not None:
                    st.session_state.pending_tool_inputs = None
                    st.session_state.tool_inputs_original_msg = None
                # Inject any staged attached file into the message
                if st.session_state.chat_attached_file:
                    msg = msg + f" [File attached: {st.session_state.chat_attached_file}]"
                    st.session_state.chat_attached_file = None
                    st.session_state.chat_attach_sig = None
                st.session_state.chat_history.append(("user", msg))
                with st.spinner("Agent is thinking…"):
                    try:
                        res = requests.post(f"{API_BASE}/chat/", json={"message": msg})
                        if res.status_code == 200:
                            data = res.json()
                            if data.get("needs_upload"):
                                st.session_state.pending_chat_upload = data["needs_upload"]
                                st.session_state.chat_upload_sig = None
                            elif data.get("needs_tool_inputs"):
                                st.session_state.pending_tool_inputs = data["needs_tool_inputs"]
                                st.session_state.tool_inputs_original_msg = msg
                            else:
                                if data.get("model_id"):
                                    st.session_state.model_id = data["model_id"]
                            st.session_state.chat_history.append(("agent", data["response"]))
                        else:
                            error_msg = res.json().get("detail", "Unknown error")
                            st.session_state.chat_history.append(("agent", f"⚠️ Error: {error_msg}"))
                    except Exception as e:
                        st.session_state.chat_history.append(("agent", f"⚠️ Exception: {str(e)}"))
            st.rerun()
    else:
        st.warning("Please configure the LLM in the sidebar to start chatting.")
        st.info("Supported providers: groq, ollama, openai, Hugging Face, gemini, llama.cpp")