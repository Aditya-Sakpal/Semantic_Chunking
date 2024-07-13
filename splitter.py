import os 
from semantic_text_splitter import TextSplitter
from llama_index.core import SimpleDirectoryReader
import re
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import uuid  
from llama_parse import LlamaParse

os.environ["OPENAI_API_KEY"] ='sk-proj-rvC8Cj2NQiWDr7ectcAmT3BlbkFJOn66hjw8tjv6dIk1OxUg'
os.environ["LLAMA_CLOUD_API_KEY"]='llx-7O0m3A0ZG46ANWroroum1yu4o9YyzNKljS53tlCaGbvVsw8G'


parser = LlamaParse(
    result_type="markdown",
    parsing_instruction="""
    Parsing Instructions :- 
        Llama Parser Extraction Instructions

        Objective: Extract only the raw text data from the provided PDF document while adhering to the following guidelines:

        Ignore Text and URLs in Images: Do not extract any text or URLs that are embedded within images.
        Include URLs in Text Format: Include URLs that are present in the document as text.
        Focus on Raw Text: Extract only the plain, raw text content from the document.
        Details:

        Ignore Text and URLs in Images: Ensure that any text or URLs that appear as part of images within the document are not included in the extracted text.
        Include URLs in Text Format: Make sure to include any URLs that are part of the text content in the document.
        Extract Plain Text: Concentrate on extracting the main text content from the PDF, maintaining the original text formatting where possible but excluding any URLs or text embedded in images
    """,
    invalidate_cache=True,
)

# use SimpleDirectoryReader to parse our file
file_extractor = {".pdf": parser}

text=''

# documents = SimpleDirectoryReader(input_files=["D:/BeyondChats/unstructured.io/pdf/When the Body Says No - The Cost of Hidden Stress2.pdf"],file_extractor=file_extractor).load_data()
documents = SimpleDirectoryReader(input_files=["Tries, Sighs, and Lullabies.txt"]).load_data()

# with open('When the Body Says No - The Cost of Hidden Stress2.txt','w', encoding='utf-8') as f :
#     for document in documents:
#         text+=document.text
#     f.write(text)


def remove_excess_whitespaces(text):
    # Replace multiple white spaces (including tabs and newlines) with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing white spaces
    text = text.strip()
    return text


for doc in documents:
    doc.text = remove_excess_whitespaces(doc.text)
    text+=doc.text
    


# Maximum number of tokens in a chunk
max_tokens = 500
splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", capacity=(100,500),overlap=0)

chunks = splitter.chunks(text)


with open('semantic1_chunks_ivf_india.txt','w', encoding='utf-8') as f:
    for i in chunks:
        f.write(i+'\n\n\n\n')
    
    
# pc = Pinecone(api_key='64e1b57c-27cf-456d-926c-2dfbe493e5ea')
# index = pc.Index('drmalpani')
# OPENAI_API_KEY = "sk-proj-rvC8Cj2NQiWDr7ectcAmT3BlbkFJOn66hjw8tjv6dIk1OxUg"
# OPENAI_ORGANIZATION = "org-kpRMeqZ5COwhxletMEWCNJCN"
# EMBEDDING_MODEL = "text-embedding-ada-002"

# client = OpenAI(
#     api_key=OPENAI_API_KEY,
#     organization=OPENAI_ORGANIZATION
# )  


# vectors = []

# for i in chunks:
#     try:
#         response = client.embeddings.create(input=i, model=EMBEDDING_MODEL)
#         vectors.append({'id':str(uuid.uuid4()),'values': response.data[0].embedding, 'metadata': {'text':i}})   
#         print(i,'Sucess') 
#     except Exception:
#         print(i,"Error in 3rd try block",Exception)
#         continue
    
    
# index.upsert(vectors=vectors, namespace="When the Body Says No - The Cost of Hidden Stress2.pdf",batch_size=100)