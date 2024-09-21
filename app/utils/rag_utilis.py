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
        separators=["\n\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(chunks, client):
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
            index.upsert(vectors=batch,namespace = namespace_name)
        
        return True  # If all operations complete successfully, return True
    
    except Exception as e:
        # If any error occurs, log the error and return False
        print(f"An error occurred during batch upsert: {str(e)}")
        return False

# def query_pinecone(pinecone_client, index_name, namespace, query_embedding):
#     index=pinecone_client.Index(index_name)
 
#     results=index.query(
#     # namespace=namespace,
#     vector=query_embedding[0],
#     top_k=3,
#     include_metadata=True,
#     # include_values=True,
#     )
#     print("Res",results)
#     # Query Pinecone for relevant vectors
#     # results = pinecone_client.query(
#     #     index_name=index_name,
#     #     namespace=namespace,
#     #     vector=query_embedding
#     # )
    
#     chunks = [match['metadata']['text'] for match in results['matches']]
#     # Extract the matching chunks (the text associated with the vectors)
 
#     # ol= [result['id'] for result in results['matches']]
    
#     return chunks

def query_pinecone(pinecone_client, index_name, namespaces, query_embedding):
    """
    Query Pinecone across multiple namespaces and retrieve relevant chunks.
    
    Parameters:
    - pinecone_client: Initialized Pinecone client.
    - index_name: The name of the Pinecone index.
    - namespaces: List of namespaces to query.
    - query_embedding: Embedding of the query to search for.

    Returns:
    - List of chunks retrieved from all namespaces.
    """
    index = pinecone_client.Index(index_name)
    
    all_chunks = []
    
    for document in namespaces:
        namespace = document.get("namespace_name")
        results = index.query(
            vector=query_embedding[0],
            top_k=5,
            include_metadata=True,
            namespace=namespace
        )
        
        # Process the results
        chunks = [match['metadata']['text'] for match in results['matches']]
        all_chunks.extend(chunks)  # Add chunks from current namespace to the list
    
    return all_chunks

previous_responses = []

def generate_answer(chunks, query, openai_client):
    # Combine the retrieved chunks and generate an answer using GPT
    context = "\n".join(chunks)
    previous_responses_str = "\n\n".join(previous_responses[-2:])
    print("context",context)
    print("query",query)
  
    prompt = f"""
    You are a knowledgable AI tasked with helping users generating a Standard Document. The user will specify his product.
    You are to ask him questions to help them focus their questions on the relevant topic. You will also be SOMETIMES provided a context from a document,
    related to the product.
        Context Information:
    {context}
    
    Previous GPT Responses:
    {str(previous_responses_str)}

    Question:
    {query}

    Guidelines for Answering:

    1. **Direct Relevance**: If the question is directly related to the context, provide a detailed answer with explanations. Organize your response under appropriate headings and subheadings to ensure clarity and structure.

    2. **Out of Context**: If the question is not relevant to the context or is significantly unrelated, respond with the following message: *"I am not trained to answer this question."* You may include a brief explanation as to why the topic does not match the context, if necessary.

    3. You are to ask him questions to help them focus their questions on the relevant topic. You will also be SOMETIMES provided a context from a document,
    related to the product.
    
    4. Be courteous
    
    5. You are allowed to use your own knowledge base if you feel the context is insufficient
    
    6. If the user has already mentioned his/her requirements, generate a report IN A WELL FORMATTED MANNER.
    
    7.IF THE USER HAS ALREADY MENTIONED his/her requirements, generate a report IN A WELL FORMATTED MANNER
    """


    completion = openai_client.chat.completions.create(
    model="gpt-4o-mini",

    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": prompt
        }
    ]
    )
    return completion.choices[0].message