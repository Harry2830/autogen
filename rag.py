import os
import autogen
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import OpenAIChatCompletionClient
import asyncio

load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

user_memory = ListMemory()

model_client = OpenAIChatCompletionClient(
    model="gpt-3.5-turbo",
    api_key=OPENAI_API_KEY,
    temperature=0,
)


# llm_config = {
#     "model": "gpt-3.5-turbo",
#     "api_key": OPENAI_API_KEY,
#     "temperature": 0,
# }

# ✅ Initialize ChromaDB Vector Store
vector_db = ChromaVectorDB(path="./chroma_db")


collection = vector_db.create_collection("rag_collection")


embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"])

assistant = AssistantAgent(
    name="Assistant",
    system_message="You are a helpful AI assistant. Use provided context to answer questions.",
    model_client=model_client,
    memory=[user_memory],
)

user_proxy = UserProxyAgent(
    name="User",
    # input_func=input()
    # code_execution_config={"use_docker": False}  # Use a dictionary here ✅
)



def retrieve_documents(query, n_results=2):
    """Retrieve relevant documents for RAG from ChromaDB."""
    results = collection.query(query_texts=[query], n_results=n_results)
    
    # Extract documents safely
    retrieved_docs = results.get("documents", [])  # Get list of docs, default to empty list

    # Ensure it's a flat list
    if retrieved_docs and isinstance(retrieved_docs[0], list):  
        retrieved_docs = [doc for sublist in retrieved_docs for doc in sublist]  # Flatten nested lists
    
    return retrieved_docs

# Prepare an initial query that uses retrieved context
query = "What is zero shot learning capabilities?"
retrieved_docs = retrieve_documents(query)
# Since our collection returns strings, simply join them
context = "\n\n".join(retrieved_docs)
final_query = f"Using the following retrieved context, answer the question:\n\n{context}\n\nQ: {query}"

# Create an asynchronous function to run the conversation interactively
async def interactive_chat():
    # Start the conversation using run_stream() which returns an async generator
    stream = assistant.run_stream(task=final_query)
    async for message in stream:
        print(message)
    
    # Now enter a loop to receive human inputs and feed them into the conversation
    while True:
        human_input = input("Enter your next query or feedback (type 'exit' to quit): ")
        if human_input.lower().strip() == "exit":
            break
        # Each new human input becomes a new task
        stream = assistant.run_stream(task=human_input)
        async for message in stream:
            print(message)

# Run the interactive chat loop
asyncio.run(interactive_chat())