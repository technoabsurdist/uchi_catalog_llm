from dotenv import load_dotenv
from langchain.chains import RetrievalQA
import chromadb
from langchain.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from prompts import uchicago_system_prompt

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

response = chain(uchicago_system_prompt)
print(response['result'])

