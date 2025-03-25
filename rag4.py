import os
import asyncio
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from pinecone import Pinecone
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.ui import Console
from autogen_core.memory import ListMemory

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

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"], chunk_size=500, chunk_overlap=50)

# Query Expansion Agent
query_expansion_agent = AssistantAgent(
    name="query_expander",
    model_client=model_client,
    system_message=(
        "You are an expert at reformulating search queries for better document retrieval. "
        "Given a user query, generate 3 semantically diverse variations of it."
    ),
    memory=[user_memory],
)

def retrieve_documents(query: str, n_results: int = 3):
    """Retrieve documents from Pinecone using agent-generated multi-query expansion."""
    
    # Step 1: Query Expansion Agent generates variations
    expansion_prompt = f"Generate 3 diverse variations of this query: {query}"
    expansion_response = query_expansion_agent.generate_reply(expansion_prompt)
    
    if not expansion_response:
        return []  # Return empty list if expansion fails

    query_variations = expansion_response.content.split("\n")

    retrieved_docs = []
    for q in query_variations:
        query_embedding = embedding_function(q)

        results = index.query(
            vector=query_embedding,
            top_k=n_results,
            include_metadata=True
        )

        docs = results['matches']
        for doc in docs:
            retrieved_docs.append(doc['metadata']['text'])

    return list(set(retrieved_docs))[:n_results]

# Retrieval Agent (Uses retrieve_documents function)
retrieval_agent = AssistantAgent(
    name="retriever",
    model_client=model_client,
    system_message="You retrieve relevant documents from Pinecone using the `retrieve_documents` tool.",
    memory=[user_memory],
    tools=[retrieve_documents],  # ✅ Pass function, not agent
)

# Assistant Agent (Requests retrieval first)
assistant = AssistantAgent(
    name="assistant",
    model_client=model_client,
    system_message=(
        "You are a RAG-powered AI assistant. "
        "First, ask the `retriever` agent to fetch relevant knowledge, then use it to answer the query."
    ),
    memory=[user_memory],
)

# Confirmation Agent
confirmation = AssistantAgent(
    name="confirmation",
    model_client=model_client,
    system_message="You approve every response from the assistant by replying 'APPROVE'.",
    memory=[user_memory],
)

# User Proxy
user_proxy = UserProxyAgent("user_proxy", input_func=input)

# Termination Condition
termination = TextMentionTermination("APPROVE")

# Sequential Team Chat (Assistant -> Retrieval -> Assistant -> Confirmation)
team = RoundRobinGroupChat([assistant, retrieval_agent, confirmation], termination_condition=termination)

async def interactive_chat():
    """Run interactive chat session asynchronously."""
    print("\n=== AI Assistant Chat with Agentic Multi-Query RAG ===")
    print("Type your message or 'exit' to end the conversation.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break

        # Assistant requests retrieval before answering
        stream = team.run_stream(task=user_input)  
        await Console(stream)

if __name__ == "__main__":
    asyncio.run(interactive_chat())
