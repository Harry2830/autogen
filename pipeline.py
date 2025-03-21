import os
import asyncio
from dotenv import load_dotenv
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent, Agent
from autogen_agentchat.teams import Pipeline
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.conditions import TextMentionTermination
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from autogen_agentchat.ui import Console

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Initialize OpenAI model client
model_client = OpenAIChatCompletionClient(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

# Initialize Vector Database
vector_db = ChromaVectorDB(path="./chroma_db")
collection = vector_db.get_collection("rag_collection")

# Define a function for document retrieval
def retrieve_documents(query, n_results=2):
    """Retrieve relevant documents from ChromaDB for context-aware responses."""
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved_docs = results.get("documents", [])

    if retrieved_docs and isinstance(retrieved_docs[0], list):  
        retrieved_docs = [doc for sublist in retrieved_docs for doc in sublist]  

    return retrieved_docs

# Query Processing Agent
query_processor = Agent(
    name="QueryProcessor",
    system_message="Your job is to clean and preprocess user queries before passing them to the retrieval system.",
    model_client=model_client,
    function=lambda query: query.strip()  # Simple query processing
)

# Document Retrieval Agent
retrieval_agent = Agent(
    name="RetrievalAgent",
    system_message="Your job is to retrieve the most relevant documents from the database.",
    model_client=model_client,
    function=lambda query: retrieve_documents(query)  # Retrieve documents
)

# Response Generation Agent
response_agent = AssistantAgent(
    name="ResponseAgent",
    model_client=model_client,
    system_message="You are an AI assistant that answers user queries using retrieved context.",
)

# User Proxy Agent (Handles user interaction)
user_proxy = UserProxyAgent("User", input_func=input)

# Termination condition: Ends conversation when user types "APPROVE"
termination = TextMentionTermination("APPROVE")

# Define the multi-agent pipeline
pipeline = Pipeline(
    agents=[query_processor, retrieval_agent, response_agent],  
    termination_condition=termination
)

async def interactive_chat():
    """Runs an interactive chat session with an automated RAG pipeline."""
    print("\n=== AI Assistant Chat ===")
    print("Type your message or 'APPROVE' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()

        # Immediate termination on "APPROVE"
        if user_input.upper() == "APPROVE":
            print("Chat session ended.")
            break  

        if not user_input:
            print("Please enter a valid query.")
            continue

        # Pass the query through the pipeline
        stream = pipeline.run_stream(task=user_input)
        await Console(stream)

if __name__ == "__main__":
    asyncio.run(interactive_chat())
