import os
import autogen
import asyncio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen.agentchat.contrib.vectordb.chromadb import ChromaVectorDB
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console


load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0.7, 
)

vector_db = ChromaVectorDB(path="./chroma_db")
collection = vector_db.get_collection("rag_collection")
embedding_function = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"])

def retrieve_documents(query, n_results=2):
    """Retrieve relevant documents from ChromaDB based on the query."""
    results = collection.query(query_texts=[query], n_results=n_results)
    retrieved_docs = results.get("documents", [])

    if retrieved_docs and isinstance(retrieved_docs[0], list):
        retrieved_docs = [doc for sublist in retrieved_docs for doc in sublist]
    
    return retrieved_docs

assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message="You are a helpful AI assistant. Use the provided context to answer questions accurately.",
)

user_proxy = UserProxyAgent("user_proxy", input_func=input)  # Uses console input for queries

termination = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat([assistant, user_proxy], termination_condition=termination)

async def interactive_chat():
    """Runs the interactive chat session asynchronously."""
    print("\n=== AI Assistant Chat ===")
    print("Type your message or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break

        retrieved_docs = retrieve_documents(user_input)
        context = "\n\n".join(retrieved_docs)
        final_query = f"Using the following retrieved context, answer the question:\n\n{context}\n\nQ: {user_input}"

        stream = team.run_stream(task=final_query)
        await Console(stream)

if __name__ == "__main__":
    asyncio.run(interactive_chat())
