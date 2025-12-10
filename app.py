# app.py
"""
Robust Gemini Agent Chatbot (Streamlit)
- Safe on_change pattern for text input
- Graceful fallback if google-genai or GEMINI_API_KEY is missing
- Local tools: todo, calculator, simple doc upload + keyword search
"""

import streamlit as st
from dotenv import load_dotenv
import os
import re
import json
import tempfile
from pathlib import Path
from typing import List, Dict

# ---------- optional imports ----------
try:
    from google import genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False

# -------------------------
# Load environment / API key
# -------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize client only if library + key available
client = None
if GENAI_AVAILABLE and GEMINI_API_KEY:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        client = None
        client_init_error = str(e)
else:
    client_init_error = None

# ---------- Status shown at top so a blank page doesn't hide errors ----------
st.set_page_config(page_title="Gemini Agent Chatbot (Robust)", layout="wide")
st.title("Gemini Agent Chatbot — Agent Mode (Robust)")

# Show environment status so you immediately see why model calls might be disabled
cols = st.columns(3)
with cols[0]:
    st.write("**Gemini client installed:**", "✅" if GENAI_AVAILABLE else "❌")
with cols[1]:
    st.write("**GEMINI_API_KEY provided:**", "✅" if bool(GEMINI_API_KEY) else "❌")
with cols[2]:
    st.write("**PDF support (PyPDF2):**", "✅" if PDF_SUPPORT else "❌")

if GENAI_AVAILABLE and not GEMINI_API_KEY:
    st.warning("GenAI library is installed but GEMINI_API_KEY is missing. Add it to .env or Streamlit Secrets.")
if not GENAI_AVAILABLE:
    st.info("google-genai library not installed — model calls will be disabled. Install `google-genai` to enable.")

if client is None and GENAI_AVAILABLE and GEMINI_API_KEY:
    st.error("Failed to initialize Gemini client. See logs for details.")
    if client_init_error:
        st.code(client_init_error)

# -------------------------
# Helpers: safe text cleaning / tokenizing
# -------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def tokenize_words(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())

# -------------------------
# Tools (local functions)
# -------------------------
if "todos" not in st.session_state:
    st.session_state.todos = []

def tool_add_todo(item: str) -> str:
    item = normalize_text(item)
    st.session_state.todos.append(item)
    return f"Added todo: {item}"

def tool_list_todos() -> str:
    if not st.session_state.todos:
        return "Your todo list is empty."
    return "Todos:\n" + "\n".join(f"- {t}" for t in st.session_state.todos)

def tool_calculate(expr: str) -> str:
    expr = expr.strip()
    # Only permit digits/operators/parentheses/decimal
    if not re.fullmatch(r"[\d\.\s\+\-\*\/\(\)]+", expr):
        return "Calculation refused: expression contains invalid characters."
    try:
        result = eval(expr, {"__builtins__": None}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# -------------------------
# Document store + search (simple)
# -------------------------
if "docs" not in st.session_state:
    st.session_state.docs = []
    st.session_state.next_doc_id = 1

def add_document(name: str, text: str) -> str:
    text = normalize_text(text)
    tokens = tokenize_words(text)
    doc = {
        "id": st.session_state.next_doc_id,
        "name": name,
        "text": text,
        "tokens": tokens
    }
    st.session_state.docs.append(doc)
    st.session_state.next_doc_id += 1
    return f"Document '{name}' uploaded (id={doc['id']})."

def search_documents(query: str, top_k: int = 3) -> List[Dict]:
    q_tokens = set(tokenize_words(query))
    scored = []
    for d in st.session_state.docs:
        shared = q_tokens.intersection(set(d["tokens"]))
        score = len(shared)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

# -------------------------
# Heuristic to decide which tool to call
# -------------------------
def detect_tool_call(user_text: str):
    text = user_text.strip()

    add_patterns = [
        r"^(?:add|put|create)\s+(?:a\s+)?todo\s+(?:to\s+)?(.+)$",
        r"^(?:remind me to|i need to|i must)\s+(.+)$",
        r"^todo[:\-]\s*(.+)$"
    ]
    for patt in add_patterns:
        m = re.search(patt, text, flags=re.IGNORECASE)
        if m:
            item = normalize_text(m.group(1))
            return "add_todo", {"item": item}

    if re.search(r"(list|show|display).*(todos|todo|to[- ]?do)", text, flags=re.IGNORECASE) or re.search(r"what.*(do i have|todos|todo)", text, flags=re.IGNORECASE):
        return "list_todos", {}

    if re.search(r"(upload|add).*(document|file|pdf|doc|notes)", text, flags=re.IGNORECASE):
        return "ask_upload", {}
    if re.search(r"(list|show).*(documents|files|notes|docs)", text, flags=re.IGNORECASE):
        return "list_docs", {}

    if re.search(r"(search|find|look up|find in).*(document|file|notes|docs)", text, flags=re.IGNORECASE):
        m = re.search(r"(?:for|about)\s+(.+)$", text, flags=re.IGNORECASE)
        query = m.group(1).strip() if m else text
        return "search_docs", {"query": query}

    math_match = re.search(r"([-()0-9+\/*\.\s]{1,120})", text)
    if math_match:
        candidate = math_match.group(1).strip()
        if re.search(r"\d", candidate) and re.search(r"[\+\-\*\/]", candidate):
            return "calculate", {"expr": candidate}

    return None, None

# -------------------------
# Core processing callback
# -------------------------
def process_user_input():
    """
    on_change callback for user_input text widget.
    Safe place to mutate st.session_state.
    """
    user_input = st.session_state.get("user_input", "")
    user_input = user_input.strip()
    if not user_input:
        st.session_state.pop("user_input", None)
        return

    # ensure chat_history exists
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "system", "content": "You are a helpful assistant (agent). Use tool results when available."}
        ]

    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # run heuristic tool
    tool_name, tool_args = detect_tool_call(user_input)
    tool_results_text = ""

    try:
        if tool_name == "add_todo":
            result = tool_add_todo(tool_args["item"])
            st.session_state.chat_history.append({"role": "tool", "tool_name": "add_todo", "content": result})
            tool_results_text += f"add_todo -> {result}\n"
        elif tool_name == "list_todos":
            result = tool_list_todos()
            st.session_state.chat_history.append({"role": "tool", "tool_name": "list_todos", "content": result})
            tool_results_text += f"list_todos -> {result}\n"
        elif tool_name == "calculate":
            result = tool_calculate(tool_args["expr"])
            st.session_state.chat_history.append({"role": "tool", "tool_name": "calculate", "content": result})
            tool_results_text += f"calculate({tool_args['expr']}) -> {result}\n"
        elif tool_name == "list_docs":
            if not st.session_state.docs:
                result = "No documents uploaded."
            else:
                result = "\n".join([f"- id={d['id']} name={d['name']}" for d in st.session_state.docs])
            st.session_state.chat_history.append({"role": "tool", "tool_name": "list_docs", "content": result})
            tool_results_text += f"list_docs -> {result}\n"
        elif tool_name == "search_docs":
            query = tool_args.get("query", user_input)
            docs = search_documents(query)
            if not docs:
                result = f"No documents matched the query: '{query}'."
            else:
                lines = []
                for d in docs:
                    snippet = d["text"][:400] + ("..." if len(d["text"]) > 400 else "")
                    lines.append(f"(id={d['id']}) {d['name']}: {snippet}")
                result = "\n\n".join(lines)
            st.session_state.chat_history.append({"role": "tool", "tool_name": "search_docs", "content": result})
            tool_results_text += f"search_docs -> {result}\n"
        elif tool_name == "ask_upload":
            result = "Please use the Document Upload on the right to add documents."
            st.session_state.chat_history.append({"role": "tool", "tool_name": "ask_upload", "content": result})
            tool_results_text += f"ask_upload -> {result}\n"
    except Exception as e:
        # tool execution failure should not crash the app
        st.session_state.chat_history.append({"role": "tool", "tool_name": "tool_error", "content": f"Tool error: {e}"})
        tool_results_text += f"tool_error -> {e}\n"

    # Build prompt for model (if available)
    history_text = ""
    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            history_text += f"System: {msg['content']}\n"
        elif msg["role"] == "user":
            history_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            history_text += f"Assistant: {msg['content']}\n"
        elif msg["role"] == "tool":
            history_text += f"Tool ({msg.get('tool_name','tool')}): {msg['content']}\n"

    if tool_results_text:
        history_text += "\nTool Results (most recent above):\n" + tool_results_text + "\n"

    prompt_text = (
        "Below is the conversation and any Tool Results. Produce a succinct assistant reply.\n\n"
        f"{history_text}\nAssistant:"
    )

    # Call Gemini only if client is available; else return a simple fallback reply
    ai_reply = None
    if client is not None:
        try:
            with st.spinner("Thinking..."):
                response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt_text)
                ai_reply = getattr(response, "text", None) or (response.to_json() if hasattr(response, "to_json") else str(response))
                if isinstance(ai_reply, (list, dict)):
                    ai_reply = json.dumps(ai_reply)
        except Exception as e:
            ai_reply = f"[Model call failed: {e}]"
    else:
        # fallback: concise reply using available tool results or canned text
        if tool_results_text:
            ai_reply = f"Tool results provided:\n{tool_results_text}\nIf you want a model-based response, install google-genai and provide GEMINI_API_KEY."
        else:
            ai_reply = "Model not available. To enable, install google-genai and set GEMINI_API_KEY. Meanwhile, I can run local tools (todo, calculate, upload docs)."

    st.session_state.chat_history.append({"role": "assistant", "content": normalize_text(ai_reply)})

    # Clear input safely in callback
    st.session_state["user_input"] = ""

# -------------------------
# UI layout
# -------------------------
left_col, mid_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.header("Actions")
    if st.button("Clear Chat & Data"):
        st.session_state.chat_history = [st.session_state.chat_history[0]] if "chat_history" in st.session_state else [{"role":"system","content":"You are a helpful assistant (agent)."}]
        st.session_state.todos = []
        st.session_state.docs = []
        st.session_state.next_doc_id = 1
        st.rerun()

    st.markdown("**Quick tool buttons (manual):**")
    if st.button("List Todos (manual)"):
        res = tool_list_todos()
        st.session_state.chat_history.append({"role":"tool","tool_name":"list_todos","content":res})
        st.rerun()
    if st.button("Show Documents (manual)"):
        if not st.session_state.docs:
            res = "No documents uploaded."
        else:
            res = "\n".join([f"- id={d['id']} name={d['name']}" for d in st.session_state.docs])
        st.session_state.chat_history.append({"role":"tool","tool_name":"list_docs","content":res})
        st.rerun()

with mid_col:
    st.header("Conversation")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role":"system","content":"You are a helpful assistant (agent)."}]

    for msg in st.session_state.chat_history:
        if msg["role"] == "system":
            continue
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        elif msg["role"] == "assistant":
            st.markdown(f"**AI:** {msg['content']}")
        elif msg["role"] == "tool":
            st.markdown(f"**[Tool Result — {msg.get('tool_name','tool')}]** {msg['content']}")

    st.divider()
    # safe input with on_change
    st.text_input("Type your message and press Enter", key="user_input", on_change=process_user_input)

with right_col:
    st.header("Documents & Tools")
    st.markdown("**Upload files (txt or PDF)**")
    uploaded = st.file_uploader("Upload documents to make them searchable by the agent", type=["txt","pdf"], accept_multiple_files=True)
    if uploaded:
        for uf in uploaded:
            name = uf.name
            text_content = ""
            if name.lower().endswith(".txt"):
                raw = uf.read()
                try:
                    text_content = raw.decode("utf-8")
                except Exception:
                    text_content = raw.decode("latin1", errors="ignore")
            elif name.lower().endswith(".pdf"):
                if not PDF_SUPPORT:
                    st.warning("PyPDF2 not installed — PDF text extraction is disabled.")
                    continue
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name
                    reader = PdfReader(tmp_path)
                    pages = []
                    for p in reader.pages:
                        try:
                            pages.append(p.extract_text() or "")
                        except Exception:
                            pages.append("")
                    text_content = "\n".join(pages)
                except Exception as e:
                    st.error(f"PDF extraction failed for {name}: {e}")
                    continue
                finally:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except Exception:
                        pass
            else:
                st.warning(f"Unsupported file type: {name}")
                continue

            if text_content.strip() == "":
                st.warning(f"No text extracted from {name}. Skipping.")
                continue

            res = add_document(name, text_content)
            st.session_state.chat_history.append({"role":"tool","tool_name":"upload_doc","content":res})
            st.success(res)
        st.rerun()

    st.markdown("---")
    st.markdown("**Search uploaded documents**")
    doc_query = st.text_input("Search docs for...", key="doc_query")
    if st.button("Search Docs"):
        if not doc_query.strip():
            st.warning("Enter a search query.")
        else:
            docs = search_documents(doc_query, top_k=5)
            if not docs:
                res = f"No documents matched the query: '{doc_query}'."
            else:
                lines = []
                for d in docs:
                    snippet = d["text"][:400] + ("..." if len(d["text"]) > 400 else "")
                    lines.append(f"(id={d['id']}) {d['name']}:\n{snippet}")
                res = "\n\n".join(lines)
            st.session_state.chat_history.append({"role":"tool","tool_name":"search_docs_manual","content":res})
            st.rerun()

    st.markdown("---")
    st.markdown("**Current uploaded documents**")
    if not st.session_state.docs:
        st.write("No documents uploaded.")
    else:
        for d in st.session_state.docs:
            st.write(f"- id={d['id']} name={d['name']} (tokens={len(d['tokens'])})")

    st.markdown("---")
    st.write("- The app runs local tools even if the model is unavailable.")
    st.write("- To enable model replies: install `google-genai` and set `GEMINI_API_KEY` in your environment.")


