import os
from pinecone.grpc import PineconeGRPC
from openai import OpenAI
from wordsegment import load, segment
import csv

load()
os.environ["PINECONE_API_KEY"] = ''
pc = PineconeGRPC(api_key='')
index = pc.Index('drmalpani')



OPENAI_API_KEY = ""
OPENAI_ORGANIZATION = ""
EMBEDDING_MODEL = "text-embedding-ada-002"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORGANIZATION
)   

query = """	
what is the smallest number that can be expressed as a sum of two cubes in two different ways
"""


response = client.embeddings.create(input=query, model=EMBEDDING_MODEL)
# response = client.embeddings.create(input="'Hector Isn't The Problem' explain this concept", model=EMBEDDING_MODEL)
result = index.query(
    namespace="ncert2",
    # namespace="The Underground History of American Education.pdf",
    vector=response.data[0].embedding,
    top_k=10,
    include_values=False,
    include_metadata=True
)
text = ''
for res in result['matches']:
    lst=[]
    print("\n\n\n",res['metadata']['text'],"\n\n\n")
    # text+=res['metadata']['text']+'\n\n\n\n'
    # for j in res['metadata']['text'].split('.'):
    #     text=' '.join(segment(j.strip()))
    #     if not text.strip():
    #         pass
    #     else:
    #         # Ensure text is not empty before attempting to capitalize
    #         if text:
    #             text = text[0].upper() + text[1:]
    #     lst.append(text)
    #     text=text[0].upper()+text[1:]
    #     lst.append(text)
    
    # print('\n\n\n','. '.join(lst),'\n\n\n')
    
# data = {"Old Chunking": text}

# # Check if file exists, create it if it doesn't
# file_exists = os.path.isfile('Old.csv')
# with open('Old.csv', 'a', newline='', encoding='utf-8') as csvfile:
#     fieldnames = ['Old Chunking']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

#     if not file_exists:
#         writer.writeheader()  # Write header only if the file is created newly

#     # Write the data row to the CSV file
#     writer.writerow(data)





























