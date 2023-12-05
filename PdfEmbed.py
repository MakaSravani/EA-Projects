from pathlib import Path
from pdf2image import convert_from_path
import pytesseract
import re
import cv2
import os
from elasticsearch import Elasticsearch
from langchain.vectorstores import ElasticsearchStore
from langchain.embeddings.openai import OpenAIEmbeddings
from getpass import getpass
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
import openai
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk



# set elastic cloud id and password
CLOUD_ID = "pdfdb:ZWFzdHVzLmF6dXJlLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyQ0MTEwMjlmMGFmMWI0M2E4YWZkYjc1ODIzODBlMjQwOCRlN2NhOTExYjhhMGQ0MzhiOGEzOWYyNTE5MmVhODkyOQ=="
CLOUD_USERNAME = "elastic"
CLOUD_PASSWORD = "Q1XevvLfGBk1dPWZJhocspG1"
CLOUD_URL = "https://pdfdb.es.eastus.azure.elastic-cloud.com"

# set OpenAI API key
# OPENAI_API_KEY = getpass("OpenAI API key")

# embeddings = OpenAIEmbeddings(openai_api_key="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")
# model_id = 'sentence-transformers/all-MiniLM-L6-v2'
# model_kwargs = {'device': 'cpu'}

# hf_embedding = HuggingFaceEmbeddings(
#     model_name=model_id,
#     model_kwargs=model_kwargs
# )

# client = Elasticsearch(
#     cloud_id=CLOUD_ID,
#     basic_auth=(CLOUD_USERNAME, CLOUD_PASSWORD)
# )

# # Successful response!
# client.info()
index_name = 'hrdocspoc'

#create client instance
vector_store = ElasticsearchStore(
    es_cloud_id=CLOUD_ID,
    es_user=CLOUD_USERNAME,
    es_password=CLOUD_PASSWORD,
    index_name= index_name,    
    # basic_auth=(CLOUD_USERNAME,CLOUD_PASSWORD)
    # index_name= "workplace_index",
    # embedding=hf_embedding
)

#successful response!
# vector_store.info()['name']

# azure OpenAI
openai.api_key = "825a0d3880054857a94fd70649196ad3"
openai.api_base = "https://cog-cdzj2obpa54em.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15'



#create index
# vector_store.indices.delete(index=index_name, ignore=[404])

# vector_store.indices.create(index='hrdocspoc',
#         mappings={
#       "dynamic": "true",
#       "_source": {
#         "enabled": "true"
#       },
#       "properties": {
#         "creationDate": {
#           "type": "date"
#         },
#         "text_vector": {
#           "type": "dense_vector",
#           "dims": 1536
#         },
#         "text": {
#           "type": "keyword"
#         },
#         "summary": {
#           "type": "text"
#         }
#       }
#     },settings={
#       "number_of_shards": 2,
#       "number_of_replicas": 0
#     })


def embed_text(text):
    vectors = openai.Embedding.create(input=text,deployment_id="embedding")
    # print('-------------------',text)
    # print(vectors)
    return vectors['data'][0]['embedding']


def index_batch(text,embeddings):

    requests = []

    for i, doc in enumerate(text):
        request = doc
        request["_op_type"] = "index"
        request["_index"] = index_name
        request["text_vector"] = embeddings[i]
        requests.append(request)
    bulk(vector_store, requests)

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text


# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

#convert pdf file pages to images
pdf_dir = Path('C:/GITHUB/Azure Projects/EA Projects/pdfFiles')
save_dir = Path('C:/GITHUB/Azure Projects/EA Projects/pdfFileImages')

documents = []
embeddings = []
request = []
actions=[]
for images in save_dir.glob('*.jpg'):
    # Step 2: Extract text using OCR
    file_path = os.path.join(save_dir, images)
    fileName = os.path.basename(file_path)
    print(fileName)
    indexid_list = [float(s) for s in re.findall(r'-?\d+\.?\d*', fileName)]
    indexid_int = int(indexid_list[0])
    indexid = str(indexid_int)

    print('---------------- id',indexid)
    img_cv = cv2.imread(file_path)
    # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
    # we need to convert from BGR to RGB format/mode:
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    text = extract_text_from_image(img_rgb)

    documents = vector_store.from_documents(
    text, 
    embeddings,
    index_name=index_name,
    es_cloud_id=CLOUD_ID,
    es_user=CLOUD_USERNAME,
    es_password=CLOUD_PASSWORD
    )
    # print(text)
    # text_splitter = CharacterTextSplitter(
    #     chunk_size=1000,      # Specify chunk size
    #     chunk_overlap=200,    # Specify chunk overlap to prevent loss of information
    # )

    # embeddings.append(embed_text(text))
    # embed_text(text)

    # index_batch(text,embeddings)

    # action= {"_index": index_name,
    #     '_op_type': 'index',        
    #     "text_vector": embeddings,        
    #     "_id": indexid        
    #     }
    # print(action)
    # actions.append(action)

    # bulk(vector_store, actions)

vector_store.indices.refresh(index=index_name)    
print("Done embedding/indexing")


def handle_query(query,n_results):
    query_vector = embed_text(query)

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    
    response = vector_store.search(
        index="workplace_index",
        body={
            "size": n_results,
            "query": script_query,
            # "query":{
            #      "match_all": {}
            #   },            
            "_source": {"includes": ["text"]}
        }
    )

    print()
    print("{} total hits.".format(response["hits"]["total"]["value"]))
    for hit in response["hits"]["hits"]:
        print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
        print(hit["_source"])
        print()
    return response

# query_response = handle_query("company values",10)



    

    # loaded_documents = vector_store.from_documents(
    # chunked_documents, 
    # hf_embedding,  
    # index_name="workplace_index",
    # es_cloud_id=CLOUD_ID,
    # es_user=CLOUD_USERNAME,
    # es_password=CLOUD_PASSWORD
    # )


    

# for pdf_file in pdf_dir.glob('*.pdf'):
#     pages = convert_from_path(pdf_file, 300, poppler_path=r'C:/GITHUB/Azure Projects/EA Projects/Release-23.11.0-0/poppler-23.11.0/Library/bin')
#     for num, page in enumerate(pages, start=1):
#         page.save(save_dir / f'{pdf_file.stem}-page{num}.jpg', 'JPEG')

# #
# def extract_text_from_image(image):
#     text = pytesseract.image_to_string(image)
#     return text

# # Create a list to store extracted text from all pages
# extracted_text = []


# for images in save_dir.glob('*.jpg'):
#     # Step 2: Extract text using OCR
#     file_path = os.path.join(save_dir, images)    
#     img_cv = cv2.imread(file_path)
#     # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
#     # we need to convert from BGR to RGB format/mode:
#     img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#     text = extract_text_from_image(img_rgb)
#     print(text)
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,      # Specify chunk size
#         chunk_overlap=200,    # Specify chunk overlap to prevent loss of information
#     )

#     docs_split = text_splitter.split_text(text)

#     documents = vector_store.from_documents(
#     docs_split, 
#     embeddings,  
#     index_name="workplace_index",
#     es_cloud_id=CLOUD_ID,
#     es_user=CLOUD_USERNAME,
#     es_password=CLOUD_PASSWORD
#     )

