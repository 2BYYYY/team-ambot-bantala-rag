import os
# For asynchronous operations
import asyncio

from typing import Any

# For GenerativeAI
from google import genai
from google.genai import types
from google.genai.types import (
    Content,
    LiveConnectConfig,
    Modality,
    Part,
)
import numpy as np
import pandas as pd

# For similarity score
from sklearn.metrics.pairwise import cosine_similarity

# For retry mechanism
from tenacity import retry, stop_after_attempt, wait_random_exponential

from google.cloud import bigquery

PROJECT_ID = "PROJECT_ID"  # Replace with your project ID
LOCATION = "LOCATION"  # Replace with your location

# Vertex AI
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
)

MODEL_ID = "MODEL_ID" # Replace with your model of choice
MODEL = (
    f"projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}"
)

TEXT_EMBEDDING_MODEL = "text-embedding-005"

async def generate_content(query: str) -> str:
    """Function to generate text content using Gemini live API.

    Args:
      query: The query to generate content for.

    Returns:
      The generated content.
    """
    config = LiveConnectConfig(response_modalities=["TEXT"])

    async with client.aio.live.connect(model=MODEL, config=config) as session:

        await session.send_client_content(
          turns=Content(
              role="user",
              parts=[Part(text=query)]
          )
        )

        # await session.send(input=query, end_of_turn=True)

        response = []
        async for message in session.receive():
            try:
                if message.text:
                    response.append(message.text)
            except AttributeError:
                pass

            if message.server_content.turn_complete:
                response = "".join(str(x) for x in response)
                return response

@retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(4))
def get_embeddings(
    embedding_client: Any, embedding_model: str, text: str
) -> list[float]:
    """
    Generate embeddings for text with retry logic for API quota management.

    Args:
        embedding_client: The client object used to generate embeddings.
        embedding_model: The name of the embedding model to use.
        text: The text for which to generate embeddings.

    Returns:
        A list of floats representing the generated embeddings. Returns None if a "RESOURCE_EXHAUSTED" error occurs.

    Raises:
        Exception: Any exception encountered during embedding generation, excluding "RESOURCE_EXHAUSTED" errors.
    """
    try:
        response = embedding_client.models.embed_content(
            model=embedding_model,
            contents=[text]
        )
        return [response.embeddings[0].values]
    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return None
        print(f"Error generating embeddings: {str(e)}")
        raise

def get_relevant_chunks(
    query: str,
    vector_db: pd.DataFrame,
    embedding_client: Any,
    embedding_model: str,
    top_k: int = 50,
) -> str:
    """
    Retrieve the most relevant document chunks for a query using similarity search. (cosine_similarity)

    Args:
        query: The search query string.
        vector_db: A pandas DataFrame containing the vectorized document chunks.
        embedding_client: The client object used to generate embeddings.
        embedding_model: The name of the embedding model to use.
        top_k: The number of most similar chunks to retrieve.
    Returns:
        A formatted string containing the top_k most relevant chunks.
    Raises:
        Exception: If any error occurs during the process (issues with the embedding client,
                   data format problems in the vector database).
    """
    try:
        query_embedding = get_embeddings(embedding_client, embedding_model, query)

        if query_embedding is None:
            return "Could not process query due to quota issues"

        similarities = [
            cosine_similarity(query_embedding, chunk_emb)[0][0]
            for chunk_emb in vector_db["embeddings"]
        ]

        top_indices = np.argsort(similarities)[-top_k:]
        relevant_chunks = vector_db.iloc[top_indices]

        context = []
        for _, row in relevant_chunks.iterrows():
            context.append(
                {
                    "embeddings": row["embeddings"],
                    "chunk_text": row["chunk_text"],
                }
            )

        return "\n\n".join(
            [
                f"{chunk['chunk_text']}"
                for chunk in context
            ]
        )

    except Exception as e:
        print(f"Error getting relevant chunks: {str(e)}")
        return "Error retrieving relevant chunks"

@retry(wait=wait_random_exponential(multiplier=1, max=120), stop=stop_after_attempt(4))
async def generate_answer(
    query: str, context: str, llm_client: Any, modality: str = "text"
) -> str:
    """
    Generate answer using LLM with retry logic for API quota management.

    Args:
        query: User query.
        context: Relevant text providing context for the query.
        llm_client: Client for accessing LLM API.
        modality: Output modality (text or audio).

    Returns:
        Generated answer.

    Raises:
        Exception: If an unexpected error occurs during the LLM call (after retry attempts are exhausted).
    """
    try:
        # If context indicates earlier quota issues, return early
        if context in [
            "Could not process query due to quota issues",
            "Error retrieving relevant chunks",
        ]:
            return "Can't Process, Quota Issues"

        prompt = f"""
        You are BB, the helpful assistant of the Bantala Chatbot.
        Use the provided context about the Kanlaon Volcano to answer questions and expound with relevant data in a warm and approachable tone.
        If the answer is not found in the context or the question is not related to volcanoes, politely explain that it’s beyond the scope of this chatbot or that the information isn’t currently available.

        Context:{context}

        Question:{query}

        Answer:"""

        if modality == "text":
            response = await generate_content(prompt)
            return response

    except Exception as e:
        if "RESOURCE_EXHAUSTED" in str(e):
            return "Can't Process, Quota Issues"
        print(f"Error generating answer: {str(e)}")
        return "Error generating answer"