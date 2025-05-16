
# Bantala RAG system

Designed to provide accurate, real-time volcanic activity information through the Bantala chatbot

🛠️ Technologies Used:

- Vertex AI – Generates embeddings for user queries

- Cloud Run Services (Function) – Handles user queries and performs the RAG workflow via HTTP-based JSON requests/responses.

- Gemini – llm model used to generate responses based on the relevant context.

- BigQuery – Where the embedded chunks is stored

⚙️ Architecture Overview:

When A user sends a question to the Bantala Chatbot, Cloud Run Service Receives the query as a JSON payload and initiates the RAG pipeline.

The query is embedded using Vertex AI embedding models.

A BigQuery SQL query is executed to retrieve all of the data to get the most relevant chunks by comparing embeddings using cosine similarity.

If relevant context is found, it's passed along with the query to Gemini, which generates a response.

The chatbot receives the response as a JSON object and displays it to the user.

![image](https://github.com/user-attachments/assets/b7e019a2-f415-46ff-b6c8-f313fcaa1bea)

🚀 Purpose:

Aims to keep users informed about volcanic activity by delivering context-aware updates. By combining real-time scraped data with llms, the system provides reliable and easy-to-understand information all built on a scalable, serverless, and cloud-native infrastructure.
