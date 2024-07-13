from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from llama_index.embeddings.openai import OpenAIEmbedding

#
loader = PyPDFLoader("1810.04805.pdf")
documents = loader.load()
#
print(len(documents))

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
#
naive_chunks = text_splitter.split_documents(documents)
for chunk in naive_chunks[10:15]:
  print(chunk.page_content+ "\n")
  
embed_model = FastEmbedEmbeddings(model_name=OpenAIEmbedding())

semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
#
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])
#
for semantic_chunk in semantic_chunks:
  if "Effect of Pre-training Tasks" in semantic_chunk.page_content:
    print("")
    print("")
    print("")
    print("")
    print("chunk",semantic_chunk.page_content)
    print("")
    print("")
    print("")
    print("")
    print(len(semantic_chunk.page_content))
