from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from prompts import system_prompt
import chromadb
import sqlite3

# init + config
load_dotenv()
chroma_client = chromadb.Client()
model_id = "gpt-3.5-turbo"
finetuned_model_id = "ft:gpt-3.5-turbo-0125:uchicago:uchi-large5:8yKA4g7Z"
llm = ChatOpenAI(model_name=model_id)

embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="data",
    embedding_function=embeddings,
    collection_name="catalog_db_pdf"
)

chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=db.as_retriever(),
                                    verbose=True)


st.set_page_config(
    page_title="college catalog helper",
    page_icon="🏛️",
)

# Streamlit app setup
col1, col2 = st.columns([1, 2.3])

with col1:
    st.image("resources/pheonixlogo2.png", width=185)

with col2:
    st.subheader("Pheonix AI")
    st.write("Welcome to Pheonix AI, the UChicago College Catalog helper! I am an LLM-powered academic advisor designed to assist you in your academic journey. Ask me any questions related to your course selections, major decisions, graduation requirements, or any other catalog-related inquiries you might have.")

st.divider()



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # generate + process bot response
    uchicago_system_prompt = f"{system_prompt} Query: \n{prompt}"
    response = chain(uchicago_system_prompt)  
    response_text = response['result']

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    st.rerun()
