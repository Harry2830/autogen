import os
import asyncio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from pinecone import Pinecone, ServerlessSpec
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console

load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Initialize OpenAI API client
openai.api_key = OPENAI_API_KEY
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",    
    api_key=OPENAI_API_KEY,
    temperature=0.7,
)
pc = Pinecone(api_key=PINECONE_API_KEY)
# Initialize Pinecone
index_name = "autogen-rag"
index = pc.Index(index_name)

# Set up the vector database using Pinecone
embedding_function = lambda text: openai.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

# Setup for text splitting
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"], chunk_size=500, chunk_overlap=50)

def expand_query(query):
    """Generate multiple variations of the query to enhance retrieval."""
    variations = [
        f"Give detailed information on {query}",
        f"Explain {query} with examples",
        f"Provide recent research on {query}",
    ]
    return variations

def retrieve_documents(query, n_results=3):
    """Retrieve documents from Pinecone using multi-query construction."""
    queries = expand_query(query)
    retrieved_docs = []
    for q in queries:
        # Generate embedding for the query
        query_embedding = embedding_function(q)
        
        # Use Pinecone's vector search to retrieve similar documents
        results = index.query(
            vector=query_embedding,  # Pass the embedding as a named argument
            top_k=n_results,         # Specify the number of results to return
            include_metadata=True    # Optionally include metadata (e.g., text)
        )
        
        # Process results from Pinecone
        docs = results['matches']
        for doc in docs:
            retrieved_docs.append(doc['metadata']['text'])  # Assuming 'text' is stored as metadata
    
    print(f"Queries: {queries}")
    print(f"Retrieved Documents: {retrieved_docs}")
    
    return list(set(retrieved_docs))[:n_results]  # Remove duplicates


# Initialize agents and group chat
assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a RAG-powered AI assistant. Use retrieved documents for accurate answers.",
)

user_proxy = UserProxyAgent("user_proxy", input_func=input)

termination = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat([assistant, user_proxy], termination_condition=termination)

async def interactive_chat():
    """Run interactive chat session asynchronously."""
    print("\n=== AI Assistant Chat with Multi-Query RAG ===")
    print("Type your message or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break

        retrieved_docs = retrieve_documents(user_input)
        context = "\n\n".join(retrieved_docs)
        final_query = f"Using retrieved context, answer the question:\n\n{context}\n\nQ: {user_input}"

        stream = team.run_stream(task=final_query)
        await Console(stream)


if __name__ == "__main__":
    asyncio.run(interactive_chat())