from dotenv import load_dotenv
import streamlit as st
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from prompts import system_prompt
import chromadb

# init + config
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


st.set_page_config(
    page_title="college catalog helper",
    page_icon="🏛️",
)

# Streamlit app setup
# st.image("phoenix_placeholder_gray.png", width=400)
# st.subheader("Pheonix.ai") 
# st.caption("Welcome to Pheonix.ai, the UChicago College Catalog Helper! I am an AI academic advisor bot designed to assist you in navigating your academic journey effectively. Please feel free to ask me any questions related to your course selections, major decisions, graduation requirements, or any other catalog-related inquiries you might have.")
# st.divider()
col1, col2 = st.columns([1, 2.3])  # Adjust the ratio as needed

with col1:
    st.image("pheonixlogo.png", width=170)

with col2:
    st.subheader("Pheonix AI")
    st.caption("Welcome to Pheonix AI, the UChicago College Catalog helper! I am an LLM-powered academic advisor designed to assist you in your academic journey. Ask me any questions related to your course selections, major decisions, graduation requirements, or any other catalog-related inquiries you might have.")

st.divider()



if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # generate + process bot response
    uchicago_system_prompt = f"{system_prompt} Query: \n{prompt}"
    response = chain(uchicago_system_prompt)  
    response_text = response['result']

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    
    st.rerun()
