from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai
from pinecone import Pinecone, ServerlessSpec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdfplumber
from tqdm import tqdm
from dotenv import load_dotenv
import os
import pinecone
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import spacy
import re
import unicodedata
from langchain_core.prompts import PromptTemplate

def load_pdf_content(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text(layout=True)
    return text

def divide_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks
def preprocess_document(text):
    
    nlp = spacy.load("en_core_web_sm")
    # Remove mentions (e.g., @username) and hashtags (e.g., #topic)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Process the text using spaCy
    doc = nlp(text)

    cleaned_tokens = []
    
    # Define a set of acceptable punctuation
    acceptable_punctuation = {'.', ',', '!', '?', '(', ')', '[', ']', '{', '}', '"', "'", '-', ':'}

    for token in doc:
        # Keep alphabetic tokens, numbers, and acceptable punctuation
        if token.is_alpha or token.like_num or token.text in acceptable_punctuation:
            cleaned_tokens.append(token.text)
    
    # Join tokens and ensure there's no extra white space
    cleaned_text = " ".join(cleaned_tokens)
    
    # Remove any accidental extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text
def get_embeddings(chunks,client):
    """
    Retrieve embeddings for a list of text chunks from OpenAI.
    """
    embeddings = []
    for chunk in tqdm(chunks, desc="Generating Embeddings"):
        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"  # Specify the embedding model
        )
        embeddings.append(response.data[0].embedding)
    return embeddings
def batch_upsert(pinecone_client, index_name, namespace_name, standard_embeddings,chunks ,batch_size=100):
    try:
        # Load the index
        index = pinecone_client.Index(index_name)

        # Generate IDs and prepare data for batch upsert
        # vectors = [(str(i), embedding) for i, embedding in enumerate(standard_embeddings)]
        vectors = [
            (str(i), embedding, {"text": chunk})
            for i, (embedding, chunk) in enumerate(zip(standard_embeddings, chunks))
        ]

        # Upsert vectors in batches
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            index.upsert(vectors=batch, namespace=namespace_name)
        
        return True  # If all operations complete successfully, return True
    
    except Exception as e:
        # If any error occurs, log the error and return False
        print(f"An error occurred during batch upsert: {str(e)}")
        return False

def get_similar_chunks(indices, chunks):
    return [chunks[i] for i in indices]

def embed_question(question,client):
    response = client.embeddings.create(
            input=question,
            model="text-embedding-ada-002"  # Specify the embedding model
        )
    return response.data[0].embedding

def get_context(question_embedding,chunks,get_similar_chunks,pc,index_name,namespace):
    
    index = pc.Index(index_name)
    # Perform the query to find the top 3 most similar vectors
    query_result = index.query(
    vector=question_embedding,  # Use the query embedding
    top_k=10,                 # Number of top results to return
    include_metadata=True,    # Include metadata in the result to identify the chunks
    namespace=namespace       # Specify the namespace you want to query
)

    # Extract the IDs of the most similar chunks
    indices = [int(match['id']) for match in query_result['matches']]

    # Retrieve the corresponding chunks
    similar_chunks = get_similar_chunks(indices,chunks)

    # Combine the content of the similar chunks into a context string
    context = "\n\n".join([chunk for chunk in similar_chunks])

    return context
def generate_response(client,context,model="gpt-4", temperature=0.8, max_tokens=1500):
    # Define the prompt template
    prompt = PromptTemplate.from_template("""INSTRUCTIONS:
    You have been given extracts of guidelines, recommendations from a DOCUMENT retrieved using RAG. These may be difficult to understand or random. YOU ARE TO DO YOUR BEST. You may be given a section number followed by a guideline. Note down the guideline as it is.
    Generate in such a format that we can compare results from another document to determine what has been missed. Return answer in labelled sections.
    \n
    Context:{context}
    \n""")
    
    # Get context using the provided function
    
    # Format the prompt with the context
    generated_prompt = prompt.format(context=context)
    
    # Create a completion request to OpenAI
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": generated_prompt}
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Return the response content
    return response.choices[0].message.content

def generate_report(client,standard_response,personal_response,model="gpt-4", temperature=0.7, max_tokens=5000):
    # Define the prompt template
    prompt = PromptTemplate.from_template("""INSTRUCTIONS:
    I will provide you two extracts. One is from a standards document and one is our own personal version. The standards \n
    document contains guidlines/recommendations etc which have been summarized from large docuemnt using RAG. The personal \n
    document is a response to the standards document. Your task is to generate a report which will tell \n
    1. Which guidelines/points/rules/recommendations have been missed in the personal document \n
    2. Which guidelines/points/rules/recommendations have been in the personal document \n
    3. A percentage of the personal document in compliance with the standards document. \n
    \n
    Standard Document extract: {standard_response}
    \n\n
    Personal Document extract: {personal_response}
    \n
    \n
    Generate a well detailed report comparing the two document's extract.
    If either context is missing reply with "Documents incompatible, please upload relevant documents"
    \n""")
    
    # Format the prompt with the context
    generated_prompt = prompt.format(
        standard_response=standard_response,
        personal_response=personal_response
    )
    
    # Create a completion request to OpenAI
    response = client.chat.completions.create(
        messages=[
            {"role": "user", "content": generated_prompt}
        ],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Return the response content
    return response.choices[0].message