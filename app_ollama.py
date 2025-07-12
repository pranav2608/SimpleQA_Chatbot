import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import ollama
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
##os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")
os.environ['LANGCHAIN_PROJECT'] = "Chatbot QA App with Ollama"

#Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are helpful assistant, please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,llm,temperature,max_tokens):
    llm = ollama(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})

    return answer

# Title of the app
st.title('Enhanced Q&A Chatbot with OpenAI')

# Sidebar settings of the UI
st.sidebar.title("Settings")


# Dropdown to create Ollama models (This models need to be downloaded in system locally)
llm = st.sidebar.selectbox("Select an OpenAI Model", ["mistral"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

# Main interface for user input
st.write("Ask any Question")
user_input = st.text_input('Query: ')

if user_input:
    response = generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")