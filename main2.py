from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
from prompts import uchicago_system_prompt

# Initialize components as in embed.py
load_dotenv()
chroma_client = chromadb.Client()
model_id = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_id)

embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="data",
    embedding_function=embeddings,
    collection_name="catalog_db"
)

chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=db.as_retriever(search_type="mmr"),
                                    verbose=True)

# Streamlit app setup
st.header("📜 UChicago College Catalog Helper", divider='grey')

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Generate response using the chain
    response = chain(uchicago_system_prompt.format(prompt=prompt))  # Adapt prompt formatting as necessary
    response_text = response['result']  # Extract text from response object

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(response_text)

    # Add assistant's response to session state
    st.session_state.messages.append({"role": "assistant", "content": response_text})
