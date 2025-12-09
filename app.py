# app.py
import streamlit as st
from dotenv import load_dotenv
from google import genai
import os
import re
import json
import tempfile
from pathlib import Path
from typing import List, Dict

# Optional PDF text extraction
try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except Exception:
    PDF_SUPPORT = False

# -------------------------
# Load API KEY
# -------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key not found! Set GEMINI_API_KEY in your .env")
    st.stop()

client = genai.Client(api_key=api_key)

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
    """
    Safely evaluate simple arithmetic expressions that only include
    digits, parentheses, decimal points, and + - * / operators.
    """
    expr = expr.strip()
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
# This is a session-scoped simple document store. Each doc: {"id","name","text","tokens"}
if "docs" not in st.session_state:
    st.session_state.docs = []  # list of dicts
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
    """
    Very simple keyword-based scoring:
    score = number of shared tokens between query and doc tokens.
    Returns top_k docs sorted by score (score >0).
    """
    q_tokens = set(tokenize_words(query))
    scored = []
    for d in st.session_state.docs:
        shared = q_tokens.intersection(set(d["tokens"]))
        score = len(shared)
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    results = [item[1] for item in scored[:top_k]]
    return results

# -------------------------
# Utility: detect tool intent (heuristics)
# -------------------------
def detect_tool_call(user_text: str):
    """
    Basic heuristics to decide if a local tool should run before contacting the model.
    Returns (tool_name, args) or (None, None).
    """
    text = user_text.strip()

    # add todo
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

    # list todos
    if re.search(r"(list|show|display).*(todos|todo|to[- ]?do)", text, flags=re.IGNORECASE) or re.search(r"what.*(do i have|todos|todo)", text, flags=re.IGNORECASE):
        return "list_todos", {}

    # upload doc or show docs
    if re.search(r"(upload|add).*(document|file|pdf|doc|notes)", text, flags=re.IGNORECASE):
        return "ask_upload", {}
    if re.search(r"(list|show).*(documents|files|notes|docs)", text, flags=re.IGNORECASE):
        return "list_docs", {}

    # search documents
    if re.search(r"(search|find|look up|find in).*(document|file|notes|docs)", text, flags=re.IGNORECASE):
        # try to extract query phrase after 'for' or 'about'
        m = re.search(r"(?:for|about)\s+(.+)$", text, flags=re.IGNORECASE)
        query = m.group(1).strip() if m else text
        return "search_docs", {"query": query}

    # arithmetic detection: look for expression with at least one operator and digit
    math_match = re.search(r"([-()0-9+\/*\.\s]{1,120})", text)
    if math_match:
        candidate = math_match.group(1).strip()
        if re.search(r"\d", candidate) and re.search(r"[\+\-\*\/]", candidate):
            return "calculate", {"expr": candidate}

    return None, None

# -------------------------
# Streamlit UI + conversation memory
# -------------------------
st.set_page_config(page_title="Gemini Agent Chatbot (Full Agent)", layout="wide")
st.title("Gemini Agent Chatbot — Agent Mode")

# conversation history as list of dicts: {"role": "system"/"user"/"assistant"/"tool", "content": "...", optional tool_name}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant (agent). When the prompt includes a "
                "'Tool Results' section, use those results and do not invent tool outputs. "
                "If an uploaded document is provided, prefer quoting it verbatim when asked for facts."
            )
        }
    ]

# Layout: left controls, center chat, right docs & tools
left_col, mid_col, right_col = st.columns([1, 2, 1])

with left_col:
    st.header("Actions")
    if st.button("Clear Chat & Data"):
        st.session_state.chat_history = [st.session_state.chat_history[0]]
        st.session_state.todos = []
        st.session_state.docs = []
        st.session_state.next_doc_id = 1
        st.experimental_rerun()

    st.markdown("**Quick tool buttons (manual):**")
    if st.button("List Todos (manual)"):
        res = tool_list_todos()
        st.session_state.chat_history.append({"role": "tool", "tool_name": "list_todos", "content": res})
        st.experimental_rerun()

    if st.button("Show Documents (manual)"):
        if not st.session_state.docs:
            res = "No documents uploaded."
        else:
            res = "\n".join([f"- id={d['id']} name={d['name']}" for d in st.session_state.docs])
        st.session_state.chat_history.append({"role": "tool", "tool_name": "list_docs", "content": res})
        st.experimental_rerun()

with mid_col:
    st.header("Conversation")
    # display chat history
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
    # input
    user_input = st.text_input("Type your message and press Enter", key="user_input")

    if user_input:
        user_input = user_input.strip()
        if user_input == "":
            st.experimental_rerun()

        # 1) append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # 2) heuristic tool detection
        tool_name, tool_args = detect_tool_call(user_input)

        tool_results_text = ""
        # If user asked to upload via text, prompt user on right side (we will handle below)
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
                # prepare a compact result summary
                lines = []
                for d in docs:
                    snippet = d["text"][:400] + ("..." if len(d["text"]) > 400 else "")
                    lines.append(f"(id={d['id']}) {d['name']}: {snippet}")
                result = "\n\n".join(lines)
            st.session_state.chat_history.append({"role": "tool", "tool_name": "search_docs", "content": result})
            tool_results_text += f"search_docs -> {result}\n"
        elif tool_name == "ask_upload":
            # we don't perform upload here; user will be asked to use file uploader on the right.
            result = "Please use the Document Upload on the right to add documents."
            st.session_state.chat_history.append({"role": "tool", "tool_name": "ask_upload", "content": result})
            tool_results_text += f"ask_upload -> {result}\n"
        else:
            # No heuristic tool fired; proceed without local tool invocation
            tool_results_text = ""

        # 3) Build prompt for Gemini: include conversation + explicit Tool Results
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
            "Below is the full conversation and any Tool Results produced by local tools. "
            "Use the Tool Results and conversation context to produce a helpful, concise assistant reply. "
            "If Tool Results fully answer the user's request, summarize them and confirm. "
            "If not, continue the conversation and propose the next steps.\n\n"
            f"{history_text}\nAssistant:"
        )

        # 4) Call Gemini model
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_text
            )
            ai_reply = getattr(response, "text", None) or (response.to_json() if hasattr(response, "to_json") else str(response))
            # if structured, convert to string
            if isinstance(ai_reply, (list, dict)):
                ai_reply = json.dumps(ai_reply)
        except Exception as e:
            ai_reply = f"Error calling Gemini model: {e}"

        # 5) Append assistant reply
        st.session_state.chat_history.append({"role": "assistant", "content": normalize_text(ai_reply)})

        # 6) Refresh the UI so we see new messages (and clear input)
        st.session_state.user_input = ""
        st.experimental_rerun()

with right_col:
    st.header("Documents & Tools")

    # Upload documents (txt and PDF)
    st.markdown("**Upload files (txt or PDF)**")
    uploaded = st.file_uploader("Upload documents to make them searchable by the agent", type=["txt", "pdf"], accept_multiple_files=True)
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
                    st.warning("PyPDF2 not installed — PDF text extraction is disabled. Install PyPDF2 in requirements to enable.")
                    continue
                # write to temp file then read with PdfReader
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
            # append a tool result to history so the model is aware of the upload
            st.session_state.chat_history.append({"role": "tool", "tool_name": "upload_doc", "content": res})
            st.success(res)
        # rerun so uploaded docs appear in the conversation
        st.experimental_rerun()

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
            st.session_state.chat_history.append({"role": "tool", "tool_name": "search_docs_manual", "content": res})
            st.experimental_rerun()

    st.markdown("---")
    st.markdown("**Current uploaded documents**")
    if not st.session_state.docs:
        st.write("No documents uploaded.")
    else:
        for d in st.session_state.docs:
            st.write(f"- id={d['id']} name={d['name']} (tokens={len(d['tokens'])})")

    st.markdown("---")
    st.markdown("**Notes**")
    st.write(
        "- This agent uses *local tools* (calculator, todo, document search) and provides their outputs "
        "to Gemini inside the prompt under a `Tool Results` section.\n"
        "- Tool logic runs locally (safe, fast). The model is instructed to rely on those results."
    )
