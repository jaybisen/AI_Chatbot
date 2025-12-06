import streamlit as st
from dotenv import load_dotenv
from google import genai
import os

# Load API KEY
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key not found!")
    st.stop()

# Initialize Gemini client
client = genai.Client(api_key=api_key)

st.title(" Gemini Chatbot with Memory")

# Initialize session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Show chat history on screen
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.write(f" **You:** {msg['content']}")
    else:
        st.write(f" **AI:** {msg['content']}")

# User input box
user_input = st.text_input("Type your message here:")

# When user sends a message
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Build a text version of the whole conversation
    history_text = ""
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            history_text += f"User: {msg['content']}\n"
        else:
            history_text += f"Assistant: {msg['content']}\n"

    # Send the whole conversation as a single string
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history_text
    )


    ai_reply = response.text

    # Add AI message to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

    # Refresh the app so chat updates
    st.rerun()
