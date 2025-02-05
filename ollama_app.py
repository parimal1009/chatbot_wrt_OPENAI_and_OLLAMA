import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama

# Load environment variables
load_dotenv()

# Ensure LANGCHAIN_API_KEY is set
api_key = os.getenv("LANGCHAIN_API_KEY")
if not api_key:
    raise ValueError("LANGCHAIN_API_KEY is not set. Please check your .env file or set it manually.")

os.environ["LANGCHAIN_API_KEY"] = api_key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries"),
    ("user", "Question: {question}")
])

# Function to generate response
def generate_response(question, model_name, temperature, max_tokens):
    llm = Ollama(model=model_name)  # Use the selected model
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Title of the app
st.title("Enhanced Q&A Chatbot With OLLAMA")

# Sidebar model selection
llm = st.sidebar.selectbox("Select Open Source Model", ["gemma:2b"])

# Response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# User input section
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(user_input, llm, temperature, max_tokens)
    st.write("Chatbot:", response)
else:
    st.write("Please provide an input.")
