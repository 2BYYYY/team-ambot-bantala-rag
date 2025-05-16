import functions_framework
import pandas as pd
import numpy as np
import asyncio
from google.cloud import bigquery
from flask import jsonify
from rag import get_relevant_chunks, generate_answer, client, TEXT_EMBEDDING_MODEL

def big_query() -> pd.DataFrame:
    # Big Query
    client_bigquery = bigquery.Client()
    SQL_QUERY = """
        INPUT YOUR SQL QUERY HERE
    """
    vector_db_mini_vertex = client_bigquery.query(SQL_QUERY).to_dataframe()
    vector_db_mini_vertex["embeddings"] = vector_db_mini_vertex["embeddings"].apply(lambda x: np.array(x).reshape(1, -1))
    return vector_db_mini_vertex

async def process_query(query):
    vector_db_mini_vertex = big_query()
    context = get_relevant_chunks(query, vector_db_mini_vertex, client, TEXT_EMBEDDING_MODEL)
    generated_answer = await generate_answer(
    query, context, client, modality="text"
    )
    return generated_answer

@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json()
    if not request_json or "query" not in request_json:
        return jsonify({"response": "no query provides"}), 400
    
    query = request_json["query"]
    print(query)
    result = asyncio.run(process_query(query)) 

    return jsonify({"response": str(result)}), 200
