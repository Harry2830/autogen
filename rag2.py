import os
import asyncio
import autogen
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",    
    api_key=OPENAI_API_KEY,
    temperature=0.7,
)   

vector_db = ChromaVectorDB(path="./chroma_db")
collection = vector_db.get_collection("rag_collection")
embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)        

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
    """Retrieve documents from ChromaDB using multi-query construction."""
    queries = expand_query(query)
    retrieved_docs = []
    for q in queries:
        results = collection.query(query_texts=[q], n_results=n_results)
        print(results)
        docs = results.get("documents", [])
        if docs and isinstance(docs[0], list):
            docs = [doc for sublist in docs for doc in sublist]
        retrieved_docs.extend(docs)
    print(queries)
    print(retrieved_docs)
    
    return list(set(retrieved_docs))[:n_results]  # Remove duplicates

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

