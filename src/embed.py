from dotenv import load_dotenv
import chromadb
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import CharacterTextSplitter

# init + config
load_dotenv()
chroma_client = chromadb.Client()
model_id = "gpt-3.5-turbo"
llm = ChatOpenAI(model_name=model_id)

# tried with text and pdf files and they both gave inferior results to markdown
raw_documents_md = UnstructuredMarkdownLoader('../rawdata/catalog.md').load()
# raw_documents_pdf = PyPDFLoader("../rawdata/catalog.pdf").load()
text_splitter = CharacterTextSplitter(chunk_size=700, chunk_overlap=80)
docs = text_splitter.split_documents(raw_documents_md)
# docs += text_splitter.split_documents(raw_documents_pdf)

chroma_db = Chroma.from_documents(
    documents=docs, 
    embedding=OpenAIEmbeddings(), 
    persist_directory="../data", 
    collection_name="catalog_db_md"
)
