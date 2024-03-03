from dotenv import load_dotenv
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# init + config
load_dotenv()
chroma_client = chromadb.Client()
model_id = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_id)
embeddings = OpenAIEmbeddings()

raw_documents = TextLoader('rawdata/catalog_data.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=80)
documents = text_splitter.split_documents(raw_documents)
chroma_db = Chroma.from_documents(
    documents=documents, 
    embedding=embeddings, 
    persist_directory="data", 
    collection_name="catalog_db_pdf"
)