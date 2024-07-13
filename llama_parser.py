import os 
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.core.node_parser import (
    SemanticDoubleMergingSplitterNodeParser,
    LanguageConfig,
)
from abbreviations_py.textes.abbreviator import fix
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
import uuid  
import re
from urllib.parse import urlparse

os.environ["OPENAI_API_KEY"] =''
os.environ["LLAMA_CLOUD_API_KEY"]=''



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

embed_model = OpenAIEmbedding()

splitter = SemanticSplitterNodeParser(
    buffer_size=1, breakpoint_percentile_threshold=40, embed_model=embed_model
)
# config = LanguageConfig(language="english", spacy_model="en_core_web_md")

# splitter = SemanticDoubleMergingSplitterNodeParser(
#     language_config=config,
#     initial_threshold=0.5,
#     include_metadata=True,
#     appending_threshold=0.4,
#     max_chunk_size=500,
# )

base_splitter = SentenceSplitter(chunk_overlap=50)

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def clean_urls(text):
    # Pattern to find URLs, including those with spaces
    url_pattern = re.compile(r'https?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F])|[ ])+')
    matches = url_pattern.findall(text)
    
    for url in matches:
        cleaned_url = url.replace(' ', '')
        if is_valid_url(cleaned_url):
            text = text.replace(url, cleaned_url + ' ')
    
    # Ensure space before \n if it follows a URL
    parts = text.split('\n')
    for i in range(len(parts) - 1):
        if parts[i].startswith('http'):
            parts[i] += ' '
    
    return ' ' + '\n'.join(parts) + ' '

abbr_dict = {
    r'\b(?:no|No)\.?\b': 'Number',
    r'\b(?:sq|Sq)\.?\s?feet\b': 'square feet',
    r'\b(?:sq|Sq)\.?\s?km\b': 'square kilometers',
    r'\b(?:sr|Sr)\.?\b': 'Senior',
    r'\b(?:jr|Jr)\.?\b': 'Junior',
    r'\b(?:dr|Dr)\.?\b': 'Doctor',
    r'\b(?:mr|Mr)\.?\b': 'Mister',
    r'\b(?:mrs|Mrs)\.?\b': 'Missus',
    r'\b(?:ms|Ms)\.?\b': 'Miss',
    r'\b(?:st|St)\.?\b': 'Street',
    r'\b(?:ave|Ave)\.?\b': 'Avenue',
    r'\b(?:inc|Inc)\.?\b': 'Incorporated',
    r'\b(?:ltd|Ltd)\.?\b': 'Limited',
    r'\b(?:co|Co)\.?\b': 'Company',
    r'\b(?:corp|Corp)\.?\b': 'Corporation',
    r'\b(?:mt|Mt)\.?\b': 'Mount',
    r'\b(?:prof|Prof)\.?\b': 'Professor',
    r'\b(?:ph\.?d|Ph\.?D)\b': 'PhD',
    r'\b(?:jan|Jan)\.?\b': 'January',
    r'\b(?:feb|Feb)\.?\b': 'February',
    r'\b(?:mar|Mar)\.?\b': 'March',
    r'\b(?:apr|Apr)\.?\b': 'April',
    r'\b(?:aug|Aug)\.?\b': 'August',
    r'\b(?:sept|Sept)\.?\b': 'September',
    r'\b(?:oct|Oct)\.?\b': 'October',
    r'\b(?:nov|Nov)\.?\b': 'November',
    r'\b(?:dec|Dec)\.?\b': 'December',
    r'\b(?:mon|Mon)\.?\b': 'Monday',
    r'\b(?:tues|Tues)\.?\b': 'Tuesday',
    r'\b(?:wed|Wed)\.?\b': 'Wednesday',
    r'\b(?:thurs|Thurs)\.?\b': 'Thursday',
    r'\b(?:fri|Fri)\.?\b': 'Friday',
    r'\b(?:sat|Sat)\.?\b': 'Saturday',
    r'\b(?:sun|Sun)\.?\b': 'Sunday',
    r'\b(?:usa|USA)\.?\b': 'United States of America',
    r'\b(?:capt|Capt)\.?\b': 'Captain',
    r'\b(?:sgt|Sgt)\.?\b': 'Sergeant',
    r'\b(?:rev|Rev)\.?\b': 'Reverend',
    r'\b(?:esq|Esq)\.?\b': 'Esquire',
    r'\b(?:hon|Hon)\.?\b': 'Honorable',
    r'\b(?:gov|Gov)\.?\b': 'Governor',
    r'\b(?:col|Col)\.?\b': 'Colonel',
    r'\b(?:gen|Gen)\.?\b': 'General',
    r'\b(?:univ|Univ)\.?\b': 'University',
    r'\b(?:dept|Dept)\.?\b': 'Department',
}

def remove_unnecessary_periods(text):
    # Replace contractions and abbreviations
    for abbr, full in abbr_dict.items():
        text = re.sub(abbr, full, text, flags=re.IGNORECASE)
    
    return ' '.join(text.split())
    # return text

def process_pdf(pdf_path):

    documents = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    # documents = SimpleDirectoryReader(input_files=[pdf_path], file_extractor=file_extractor).load_data()

    
    for doc in documents:
        try:
            doc.text = remove_unnecessary_periods(doc.text)
            print("\n\n\nDoc text:-",doc.text,"\n\n\n")
        except Exception:
            continue
        
    nodes = splitter.get_nodes_from_documents(documents)
    
    # Split documents into nodes using SentenceSplitter
    base_nodes = base_splitter.get_nodes_from_documents(documents)
    
    # Add metadata to each node
    for node in nodes:
        try:
            # pattern = re.compile(r'page_label:\s*(\d+)')
            # match = pattern.search(node.get_metadata_str())
            content = node.get_content()
            cleaned_content = clean_urls(content)
            node.metadata = {
                "source": pdf_path,
                "text": cleaned_content,
                # "page": match.group(1) if match else None
            }
            node.text = cleaned_content
        except Exception:
            print("Error in 1st try block")
            continue
        
    return nodes, base_nodes


lst = []

pdf_paths = [
    "medical_apprecials_data.pdf"
    # "giant_panda_factsheet2006.pdf",
    # "ET-9-Project-Tiger.pdf",
    # "Lion-factsheet-on-Arkive-Panthera-leo.pdf",
    # "wolf.pdf"
]

for pdf_path in pdf_paths:
    chunks = []
    nodes, base_nodes = process_pdf(pdf_path)

    for node in nodes:
        try:
            content = node.get_content()
            chunks.append({
                "text":  clean_urls(content),
                "metadata": node.metadata
            })
        except Exception:
            print("Error in 2nd try block")
            continue
    lst.append(chunks)



pc = Pinecone(api_key='')
index = pc.Index('')
OPENAI_API_KEY = ""
OPENAI_ORGANIZATION = ""
EMBEDDING_MODEL = "text-embedding-ada-002"

client = OpenAI(
    api_key=OPENAI_API_KEY,
    organization=OPENAI_ORGANIZATION
)  

for i in lst[0]:
    print("\n\n\n",i['metadata'],"\n\n\n")

# vectors = []
# for i in lst[0]:
#     try:
#         response = client.embeddings.create(input=i['text'], model=EMBEDDING_MODEL)
#         vectors.append({'id':str(uuid.uuid4()),'values': response.data[0].embedding, 'metadata': i['metadata'] })    
#     except Exception:
#         print("Error in 3rd try block")
#         continue
    
    
# index.upsert(vectors=vectors, namespace="aditya",batch_size=100)










































