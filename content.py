from dotenv import load_dotenv
import os

# Load the .env file
load_dotenv('.env')

# Access environment variables
PINECONE_API_KEY= os.getenv('PINECONE_API_KEY')
PINECONE_API_ENVIRONMENT = os.getenv('PINECONE_API_ENVIRONMENT')
HF_TOKEN = os.getenv('HF_TOKEN')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import pdfplumber
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

import pdfplumber

# Function to load PDF using pdfplumber
def load_pdf_with_pdfplumber(file_path):
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            documents.append(page.extract_text())
    return documents

# Path to the PDF file (adjust this path if the PDF is inside a subfolder)
pdf_path = "/workspaces/codespaces-jupyter/data/memory/introduction to computers  notes.pdf"

# Load PDF and extract text
pdf_text = load_pdf_with_pdfplumber(pdf_path)

# Print extracted text (for verification)
#print("\n".join(pdf_text))


def text_split(pdf_text):
    # Wrap the raw text into a Document object
    documents = [Document(page_content=text) for text in pdf_text]
    
    # Initialize a text splitter (for example, splitting by character)
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=50)
    
    # Split the documents into chunks
    text_chunks = text_splitter.split_documents(documents)
    
    return text_chunks

result=text_split(pdf_text)
#print(len(result))

def embeding_my_text(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") 
    text_list = [doc.page_content for doc in text_chunks]
    #embeded_text=embeddings.embed_query(text_list)
    # Embed the text (using embed_documents to handle multiple documents)
    embeded_text = embeddings.embed_documents(text_list)
    return embeded_text
embeding_my_text(result)
#STORING MY EMBEDDING IN MY PINE CONE DATA BASE
embeded_result = embeding_my_text(result)

# Initialize Pinecone with the correct method
pc = Pinecone(Api_key=PINECONE_API_KEY)

# Create an index if it doesn't exist
index_name = "fresh"

# Connect to the Pinecone index
index = pc.Index(index_name)

# Assuming you have the text chunks and their IDs (document IDs)
document_ids = [f"doc_{i+1}" for i in range(len(result))]   # You should have unique IDs for each document


# Prepare the data to be upserted into Pinecone (IDs and embeddings)
vectors_to_upsert = [
    {"id": f"doc_{i+1}", "values": embed, "metadata": {"content": text}}
    for i, (embed, text) in enumerate(zip(embeded_result, [doc.page_content for doc in result]))
]

# Upsert the embeddings into Pinecone
index.upsert(vectors=vectors_to_upsert)


#LOOKING FOR SIMILAR SEARCHES
def query_pinecone(question):
    # Initialize HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Generate embedding for the question
    question_embedding = embeddings.embed_query(question)
    
    # Ensure the query vector is a list of floats
    if not isinstance(question_embedding, list):
        question_embedding = question_embedding.tolist()
    
    # Query Pinecone with the generated vector
    query_result = index.query(vector=question_embedding, top_k=1, include_metadata=True)  # Fetch top 1 result with metadata
    
    # Check if matches are returned
    if query_result['matches']:
        most_similar_doc_id = query_result['matches'][0]['id']
        similarity_score = query_result['matches'][0]['score']
        content = query_result['matches'][0]['metadata'].get('content', 'No content available')
        
        # Clean up formatting (optional)
        content_cleaned = ' '.join(content.split()).strip()  # Remove extra spaces and newlines
        
        return most_similar_doc_id, similarity_score, content_cleaned
    else:
        return None, None, "No similar document found."

# Example usage
question = input("answer: ")
doc_id, score, answer = query_pinecone(question)


prompt_template1="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {answer}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
           prompt_template1 ,
        ),
        ("human", question),
    ]
)

chain = prompt | llm
result=chain.invoke(
    {
        "answer": answer,
        "question": question,
    
    }
)
print(result.content)