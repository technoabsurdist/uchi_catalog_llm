from dotenv import load_dotenv
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from prompts import system_prompt
import chromadb

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# init + config
load_dotenv()
chroma_client = chromadb.Client()
model_id = "gpt-3.5-turbo"
embeddings = OpenAIEmbeddings()
db = Chroma(
    persist_directory="data",
    embedding_function=embeddings,
    collection_name="catalog_db_md"
)


st.set_page_config(
    page_title="college catalog helper",
    page_icon="🏛️",
)

col1, col2 = st.columns([0.5, 1.6])

with col1:
    st.image("../resources/phoenixlogo.svg", width=130)

with col2:
    st.subheader("Pheonix AI")
    st.caption("Welcome to Pheonix AI, the AI-powered UChicago Academic Advisor! I am designed to assist you in your academic journey. Ask me questions related to your courses, major decisions, graduation requirements, or any other catalog-related inquiries you have.")

st.divider()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything!"):
    # append user input to state
    st.session_state.messages.append({"role": "user", "content": prompt})

    # create response
    model = ChatOpenAI()
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(prompt)
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])
    system_prompt = f"""{system_prompt}.\n\nAnswer the question based on the following context:

    {format_docs(docs)}

    Question: {prompt}
    """
    response = model.invoke(system_prompt)

    # append model response to state
    st.session_state.messages.append({"role": "assistant", "content": response.content})
    st.rerun()
