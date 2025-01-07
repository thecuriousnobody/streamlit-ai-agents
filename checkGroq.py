from groq import Groq
import os
import sys
import streamlit as st
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
groq_api_key = st.secrets["GROQ_API_KEY"]

client = Groq(
    api_key=groq_api_key,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "what are the theories behind how the grand canyon was formed",
        }
    ],
    model = "llama-3.3-70b-versatile"
)

print(chat_completion.choices[0].message.content)