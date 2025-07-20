import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables
load_dotenv()
ollama_api_key = os.getenv("OLLAMA_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI - Title and Sidebar
st.title("Welcome to Enhanced Q&A Chatbot with OpenAI, Groq, or Ollama")
st.sidebar.title("Model Configuration")

# API Key input
api_key = st.sidebar.text_input("Enter your API Key (OpenAI or Groq)", type="password")

# LLM selection
llm_options = ["gpt-3.5-turbo", "gpt-4", "groq-1", "mistral", "gemma:2b"]
llm = st.sidebar.selectbox("Select LLM", llm_options, index=0)

# Temperature & Max Tokens
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
max_tokens = st.sidebar.slider("Max Tokens", 1, 2048, 1000)

# User Input
st.write("Ask a question to me:")
user_input = st.text_input("Your Question", placeholder="Type your question here...")

# Generate response function
def generate_response(question, api_key, llm, temperature, max_tokens, chat_history):
    # Select LLM instance
    if llm.startswith("gpt"):
        os.environ["OPENAI_API_KEY"] = api_key
        llm_instance = ChatOpenAI(model=llm, temperature=temperature, max_tokens=max_tokens)
    elif llm.startswith("groq"):
        llm_instance = ChatGroq(model="gemma2-9b-it", api_key=api_key, temperature=temperature, max_tokens=max_tokens)
    else:  # Assume Ollama
        llm_instance = Ollama(model=llm)

    # Compose chat prompt
    prompt_messages = [("system", "You are a helpful assistant. Please answer the question as best you can.")]
    for msg in chat_history:
        if isinstance(msg, HumanMessage):
            prompt_messages.append(("user", msg.content))
        elif isinstance(msg, AIMessage):
            prompt_messages.append(("ai", msg.content))
    prompt_messages.append(("user", question))

    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    output_parser = StrOutputParser()
    chain = prompt | llm_instance | output_parser
    answer = chain.invoke({"question": question})
    return answer

# Handle generate button
if st.button("Generate Response"):
    if user_input:
        with st.spinner("Generating response..."):
            try:
                response = generate_response(
                    user_input,
                    api_key,
                    llm,
                    temperature,
                    max_tokens,
                    st.session_state.chat_history
                )
                # Append to chat history
                st.session_state.chat_history.append(HumanMessage(content=user_input))
                st.session_state.chat_history.append(AIMessage(content=response))

                st.success(f"Response: {response}")
            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question before generating a response.")

# Chat History Viewer
st.markdown("### Chat History")
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        st.markdown(f"**You:** {msg.content}")
    elif isinstance(msg, AIMessage):
        st.markdown(f"**AI:** {msg.content}")

# Optional: Clear history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared.")
