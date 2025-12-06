# 🤖 Gemini AI Chatbot with Memory (Streamlit)

This project is a conversational AI chatbot built using **Google Gemini**, **Streamlit**, and **Python**.  
The chatbot supports **multi-turn conversation** using Streamlit's `session_state` and can remember previous messages to generate context-aware replies.

---

## 📌 Features

- 💬 **Conversational Memory** (Chatbot remembers past interactions)
- ⚡ **Powered by Gemini 2.5 Flash Model**
- 🌐 **Streamlit Web Interface**
- 🔐 **Secure API Key Handling using .env file**
- 🔄 **Automatically updates chat on each message**
- 🧹 **Clear Chat History Button**

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Google Gemini (google-genai SDK)**
- **python-dotenv** (to load API keys)

---

## 📂 Project Structure

your-project-folder/
│
├── app.py # Main Streamlit application
├── requirements.txt # Required dependencies
└── .env # Contains GEMINI_API_KEY (not uploaded)

## 🔧 Installation & Setup

1. **Clone the repository**
git clone https://github.com/jaybisen/AI_Chatbot
cd your-repo-name

2.Install dependencies
pip install -r requirements.txt

3.Create a .env file
GEMINI_API_KEY=your_api_key_here

4.Run the Streamlit app
streamlit run app.py

🚀 Deployment (Streamlit Cloud)
Push your code to GitHub
Go to https://share.streamlit.io
Connect your GitHub repo
Select app.py and deploy
Add GEMINI_API_KEY in Streamlit Secrets
(Settings → Secrets → Add secret)

🧠 How the Chatbot Works (Short Explanation)
The user sends a message
The message is stored in st.session_state["chat_history"]
The entire conversation history is converted into a single text block
This text is sent to the Gemini model
Gemini responds based on the context
The response is added back into chat history
Streamlit re-runs and updates the UI

📚 Requirements
nginx
Copy code
streamlit
python-dotenv
google-genai
✨ Future Improvements
Add chat bubbles (UI enhancement)

Add voice input/output

Add RAG (Retrieval-Augmented Generation)

Add conversation export feature

🧑‍💻 Author
Created by Jay
Feel free to connect or share suggestions 😊
