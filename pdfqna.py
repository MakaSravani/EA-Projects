# from langchain.vectorstores import Elasticsearch
from elasticsearch import Elasticsearch
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader

#unstructured
from unstructured.cleaners.core import clean, clean_ordered_bullets
from unstructured.staging.base import convert_to_dataframe
from unstructured.documents.elements import Title, NarrativeText
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title


import openai
import os
import cv2
import pytesseract
import pandas as pd


# elastic cloud id and password
CLOUD_ID = "pdfdb:ZWFzdHVzLmF6dXJlLmVsYXN0aWMtY2xvdWQuY29tOjQ0MyQ0MTEwMjlmMGFmMWI0M2E4YWZkYjc1ODIzODBlMjQwOCRlN2NhOTExYjhhMGQ0MzhiOGEzOWYyNTE5MmVhODkyOQ=="
CLOUD_USERNAME = "elastic"
CLOUD_PASSWORD = "Q1XevvLfGBk1dPWZJhocspG1"
CLOUD_URL = "https://pdfdb.es.eastus.azure.elastic-cloud.com"

# azure OpenAI
openai.api_key = "825a0d3880054857a94fd70649196ad3"
openai.api_base = "https://cog-cdzj2obpa54em.openai.azure.com/" # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
openai.api_version = '2023-05-15'


indexname = 'hrdocspoc'
vectorfieldname='vectorfield'
pdf_dir = Path('C:/GITHUB/Azure Projects/EA Projects/pdfFiles/Contoso_employee_handbook.pdf')
save_dir = Path('C:/GITHUB/Azure Projects/EA Projects/pdfFileImages')

elements = partition(filename="C:/GITHUB/Azure Projects/EA Projects/pdfFiles/Contoso_employee_handbook.pdf")
chunks = chunk_by_title(elements, multipage_sections=True)

for chunk in chunks:
    print(chunk)
    


# elements_df = convert_to_dataframe(elements)
# print(elements_df)
# print("\n\n".join([str(el) for el in loader][:1]))
# for el in loader:

    
        
    




# pages = loader.load_and_split()
# print(pages)


# for page in pages:
#         page_content_clean_step1 = clean(page.page_content,bullets=True)
#         page_content_clean_step2 = clean_ordered_bullets(page_content_clean_step1)
        # elements = [Title(text=page.page_content),NarrativeText(text=page.page_content)]
        # print(page)
        # df= convert_to_dataframe(elements)
        

        


    # print(str(page.metadata["category"]) + ":", page.page_content[:300])


#create elastic search store client instance
# vector_store = Elasticsearch(
#     cloud_id=CLOUD_ID,
#     basic_auth=(CLOUD_USERNAME, CLOUD_PASSWORD)
# )

# Check if the index exists before deletion (optional)
# if vector_store.indices.exists(index=indexname):
#     # Delete the index
#     vector_store.indices.delete(index=indexname)
#     print(f"Index '{indexname}' deleted successfully.")
# else:
#     print(f"Index '{indexname}' does not exist.")

# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# def extract_text_from_image(image):
#     text = pytesseract.image_to_string(image)
#     return text

# # TODO - convert pdfs files to individual images ()

# # Read text from images
# for images in save_dir.glob('*.jpg'):
#     # Step 2: Extract text using OCR
#     file_path = os.path.join(save_dir, images)
#     img_cv = cv2.imread(file_path)
#     # By default OpenCV stores images in BGR format and since pytesseract assumes RGB format,
#     # we need to convert from BGR to RGB format/mode:
#     img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#     text = extract_text_from_image(img_rgb)
#     # print(text)

#     text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=200,      # Specify chunk size
#         chunk_overlap=100,    # Specify chunk overlap to prevent loss of information
#     )
#     # print(text_splitter.split_text(text))

#     text_split = text_splitter.split_text(text)    
#     # print(text_split)

#     # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     response = openai.Embedding.create(input=text_split,deployment_id="embedding")
#     print(response['data'])
#     embeddings = response['data'][0]['embedding']
#     # print(embeddings)
#     # print(text_split)
#     # item_text = dict.fromkeys(text_split).items()
#     # item_text_embeddings = dict.fromkeys(embeddings).items()
#     # print(item_text_embeddings)

#     vectorstoredocuments = vector_store.index(
#             index=indexname,
#             body= {'text' : text_split, 'embedding': embeddings},
#             id=file_path            
#     )

# #  perform search
# def generate_query_embedding(query_text):
#     # Use Azure OpenAI or your embedding method to generate query embedding
#     # Example using Azure OpenAI
#     text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#         chunk_size=1000,      # Specify chunk size
#         chunk_overlap=200,    # Specify chunk overlap to prevent loss of information
#     )
#     # print(text_splitter.split_text(text))

#     text_split = text_splitter.split_text(query_text)    
#     # print(text_split)
#     response = openai.Embedding.create(input=text_split,deployment_id="embedding")
#     return response['data'][0]['embedding']

# def semantic_search(query_embedding, index_name, vector_field, top_n=1):
#     search_body = {
#         "query": {
#             "script_score": {
#                 "query": {"match_all": {}},
#                 "script": {
#                     "source": "cosineSimilarity(params.query_vector, '{}') + 1.0".format(vector_field),
#                     "params": {"query_vector": query_embedding}
#                 }
#             }
#         },
#         "size": top_n
#     }

#     search_results = vector_store.search(index=index_name, body=search_body)
#     for hit in search_results["hits"]["hits"]:
#         print("id: {}, score: {}".format(hit["_id"], hit["_score"]))
#         print(hit["_source"]["text"])
#         print()
#     return search_results['hits']['hits']


# # Example query
# query_text = "are the performance reviews conducted annually"
# query_embedding = generate_query_embedding(query_text)

# index_name = indexname
# vector_field = 'embedding'

# search_results = semantic_search(query_embedding, index_name, vector_field)



