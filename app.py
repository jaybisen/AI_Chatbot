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
    results = [item[1] for item in scored[:top_k]]
    return results

# -------------------------
# Utility: detect tool intent (heuristics)
# -------------------------
def detect_tool_call(user_text: str):
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
        m = re.search(r"(?:for|about)\s+(.+)$", text, flags=re.IGNORECASE)
        query = m.group(1).strip() if m else text
        return "search_docs", {"query": query}

    # arithmetic detection
    math_match = re.search(r"([-()0-9+\/*\.\s]{1,120})", text)
    if math_match:
        candidate = math_match.group(1).strip()
        if re.search(r"\d", candidate) and re.search(r"[\+\-\*\/]", candidate):
            return "calculate", {"expr": candidate}

    return None, None

# -------------------------
# Core processing callback (Safe place to mutate session_state)
# -------------------------
def process_user_input():
    """
    This function is invoked by the text_input on_change callback.
    It reads st.session_state['user_input'], processes it (tools + Gemini),
    appends results to chat_history, clears the 'user_input' key, and returns.
    """
    # read the input from session state (guaranteed to exist when callback runs)
    user_input = st.session_state.get("user_input", "")
    user_input = user_input.strip()
    if not user_input:
        # clear and exit
        st.session_state.pop("user_input", None)
        return

    # Ensure chat_history exists
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

    # 1) Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 2) Heuristic tool
