from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.llms import OpenAI
import openai

CLOUD_ID = "pdfdb:ZWFzdHVzLmF6dXJlLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyQ0MTEwMjlmMGFmMWI0M2E4YWZkYjc1ODIzODBlMjQwOCRlN2NhOTExYjhhMGQ0MzhiOGEzOWYyNTE5MmVhODkyOQ=="
CLOUD_USERNAME = "elastic"
CLOUD_PASSWORD = "Q1XevvLfGBk1dPWZJhocspG1"
CLOUD_URL = "https://pdfdb.es.eastus.azure.elastic-cloud.com"

model_id = 'sentence-transformers/all-MiniLM-L6-v2'
model_kwargs = {'device': 'cpu'}

hf_embedding = HuggingFaceEmbeddings(
    model_name=model_id,
    model_kwargs=model_kwargs
)


vector_store = ElasticsearchStore(
    es_cloud_id=CLOUD_ID,    
    es_user=CLOUD_USERNAME,
    es_password=CLOUD_PASSWORD,
    index_name= "workplace_index",
    embedding=hf_embedding
)


retriever = vector_store.as_retriever()

openai.api_key = "825a0d3880054857a94fd70649196ad3"
openai.api_base = "https://cog-cdzj2obpa54em.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15'

llm = openai.ChatCompletion.create(engine="chat") #Model name: gpt-35-turbo-16k

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Be as verbose and educational in your response as possible. 
    
    context: {context}
    Question: "{question}"
    Answer:
    """
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ANSWER_PROMPT
    | llm
    | StrOutputParser()
)

ans = chain.invoke("what are company values?")

print("---- Answer ----")
print(ans)