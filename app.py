import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
##os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")
os.environ['LANGCHAIN_PROJECT'] = "Chatbot QA App with OpenAI"

#Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are helpful assistant, please respond to the user queries"),
        ("user","Question:{question}")
    ]
)

def generate_response(question,api_key,llm,temperature,max_tokens):
    openai.api_key = api_key
    llm = ChatOpenAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question':question})

    return answer

# Title of the app
st.title('Enhanced Q&A Chatbot with OpenAI')

# Sidebar settings of the UI
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key: ", type="password")

# Dropdown to create OpenAI models
llm = st.sidebar.selectbox("Select an OpenAI Model", ["gpt-4.1","o4-mini","gpt-4o-mini"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7)
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150)

# Main interface for user input
st.write("Ask any Question")
user_input = st.text_input('Query: ')

if user_input:
    response = generate_response(user_input,api_key,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")