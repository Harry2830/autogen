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
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

openai.api_key = OPENAI_API_KEY
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",    
    api_key=OPENAI_API_KEY,
    temperature=0.7,
)
user_memory = ListMemory()
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "autogen-rag"
index = pc.Index(index_name)

embedding_function = lambda text: openai.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding

def generate_representations(query, assistant):
    """Use Autogen's AssistantAgent to generate multiple representations of a query for multi-representation indexing."""
    prompt = f"Generate different representations of the following query: {query}. Provide a summary, a technical perspective, and real-world applications."
    response = assistant.generate_response(prompt)
    return response.split("\n")

def expand_query(query, assistant):
    """Generate multiple variations of the query for multi-query retrieval using AssistantAgent."""
    base_variations = [
        query,
        f"Detailed explanation of {query}",
        f"Recent research about {query}",
        f"{query} in different contexts",
        f"{query} explained with examples"
    ]
    
    representation_variations = []
    for q in base_variations:
        representation_variations.extend(generate_representations(q, assistant))
    
    return list(set(representation_variations))

def retrieve_documents(query, assistant, n_results=5):
    """Retrieve documents using multi-query and AssistantAgent-powered multi-representation indexing."""
    queries = expand_query(query, assistant)
    retrieved_docs = set()
    
    for q in queries:
        query_embedding = embedding_function(q)
        
        results = index.query(
            vector=query_embedding,  
            top_k=n_results,        
            include_metadata=True   
        )
        
        for doc in results['matches']:
            retrieved_docs.add(doc['metadata']['text'])
    
    return list(retrieved_docs)[:n_results] 

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a RAG-powered AI assistant using multi-query and multi-representation retrieval. Use retrieved documents for accurate answers.",
    memory=[user_memory],
)

confirmation = AssistantAgent(
    name="confirmation",
    model_client=model_client,
    system_message="You are an AI assistant that approves the assistant's response by saying 'APPROVE'.",
    memory=[user_memory],
)

user_proxy = UserProxyAgent("user_proxy", input_func=input)
termination = TextMentionTermination("APPROVE")
team = RoundRobinGroupChat([assistant, confirmation], termination_condition=termination)

async def interactive_chat():
    """Run interactive chat session asynchronously."""
    print("\n=== AI Assistant Chat with Multi-Query & AssistantAgent-Powered Multi-Representation RAG ===")
    print("Type your message or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break

        retrieved_docs = retrieve_documents(user_input, assistant)
        context = "\n\n".join(retrieved_docs)
        final_query = f"Using retrieved context, answer the question:\n\n{context}\n\nQ: {user_input}"

        stream = team.run_stream(task=final_query)  
        await Console(stream)

if __name__ == "__main__":
    asyncio.run(interactive_chat())
