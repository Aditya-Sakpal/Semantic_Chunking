from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.embeddings.openai import OpenAIEmbedding
import os
import re
from urllib.parse import urlparse
from llama_index.core import Document
from pinecone.grpc import PineconeGRPC as Pinecone
from llama_index.core.node_parser import SemanticSplitterNodeParser
from openai import OpenAI
import uuid  # To generate unique IDs
from pinecone.grpc import PineconeGRPC as Pinecone

os.environ["OPENAI_API_KEY"] ='sk-proj-rvC8Cj2NQiWDr7ectcAmT3BlbkFJOn66hjw8tjv6dIk1OxUg'

embed_model = OpenAIEmbedding()
splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
)

# Also baseline splitter
base_splitter = SentenceSplitter()

# List of PDF paths
pdf_paths = [
    # "giant_panda_factsheet2006.pdf",
    # "ET-9-Project-Tiger.pdf",
    # "Lion-factsheet-on-Arkive-Panthera-leo.pdf",
    "When the Body Says No - The Cost of Hidden Stress ( PDFDrive ).pdf",
    # "wolf.pdf"
]

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def clean_urls(text):
    # Pattern to find URLs, including those with spaces
    url_pattern = re.compile(r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F])|[ ])+')
    matches = url_pattern.findall(text)
    
    for url in matches:
        cleaned_url = url.replace(' ', '')
        if is_valid_url(cleaned_url):
            text = text.replace(url, cleaned_url)
    
    return ' ' + text + ' '

# Function to process each PDF and return nodes
def process_pdf(pdf_path):
    # Load the PDF document
    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    print(f"Loaded {len(documents)} documents from {pdf_path}")

    # Split documents into nodes using SemanticSplitterNodeParser
    nodes = splitter.get_nodes_from_documents(documents)

    # Split documents into nodes using SentenceSplitter
    base_nodes = base_splitter.get_nodes_from_documents(documents)
    
    # Add metadata to each node
    for node in nodes:
        node.metadata = {
            "source": pdf_path,
            "text": clean_urls(node.get_content()),
            "page": node.metadata.get("page_number", "unknown")
        }
        node.text = clean_urls(node.get_content())
    
    return nodes, base_nodes

# lst = []
# for pdf_path in pdf_paths:
#     chunks = []
#     nodes, base_nodes = process_pdf(pdf_path)
    
#     for node in nodes:
#         chunks.append({
#             "text": node.get_content(),
#             "metadata": node.metadata
#         })
#     lst.append(chunks)

# print(lst[0][0]['metadata'])

# documents = [Document(page_content=d['text'], metadata=d['metadata']) for d in lst[0]]

pc = Pinecone(api_key='64e1b57c-27cf-456d-926c-2dfbe493e5ea')
index = pc.Index('drmalpani')
OPENAI_API_KEY = "sk-proj-rvC8Cj2NQiWDr7ectcAmT3BlbkFJOn66hjw8tjv6dIk1OxUg"
OPENAI_ORGANIZATION = "org-kpRMeqZ5COwhxletMEWCNJCN"
EMBEDDING_MODEL = "text-embedding-ada-002"

index.delete(namespace="ncert3",delete_all=True)

# client = OpenAI(
#     api_key=OPENAI_API_KEY,
#     organization=OPENAI_ORGANIZATION
# )   

# vectors = []
# for i in lst[0]:
#     response = client.embeddings.create(input=i['text'], model=EMBEDDING_MODEL)
#     vectors.append({'id':str(uuid.uuid4()),'values': response.data[0].embedding, 'metadata': i['metadata'] })    



# def get_openai_embeddings(texts):
#     embeddings = []
#     for text in texts:
#         response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
#         embeddings.append(response.data[0].embedding)

# def get_openai_embeddings(texts):
#     embeddings = []
#     for text in texts:
#         response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
#         embeddings.append(response.data[0].embedding)
#     return embeddings


# texts = [doc.get_content() for doc in documents]

# embeddings = get_openai_embeddings(texts)
# embedded_documents = [{'id': str(uuid.uuid4()), 'values': embedding} for embedding in embeddings]


# print(embedded_documents[0])
# index.upsert(vectors=vectors, namespace="aditya")