import asyncio
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
import os
from dotenv import load_dotenv
from mem0 import MemoryClient, Memory

load_dotenv()
MEM0_API_KEY = os.getenv("MEM0_API_KEY")

client = MemoryClient(api_key=MEM0_API_KEY)

def get_weather(city: str) -> str:
    """Fetch current temperature for a given city."""
    weather_data = {
        "new york": "75°F",
        "los angeles": "85°F",
        "london": "55°F"
    }
    return weather_data.get(city.lower(), "Unknown city")

def check_weather(temperature: str) -> str:
    """Evaluate if the weather is suitable for an outdoor activity based on temperature."""
    try:
        temperature_value = float(temperature.replace("°F", "").replace("°C", ""))
    except ValueError:
        return "Invalid temperature format."
    
    if temperature_value > 80:
        return "The weather is hot. It's perfect for outdoor activities!"
    elif temperature_value > 60:
        return "The weather is mild. You can go out comfortably."
    else:
        return "The weather is cold. You might want to stay indoors or bundle up."

async def main() -> None:
    model_client = OpenAIChatCompletionClient(model="gpt-4o", seed=42, temperature=0)
    
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[get_weather, check_weather],
        reflect_on_tool_use=True,
        system_message="You are a helpful assistant. Use the provided functions as needed to answer the question. Reply with TERMINATE when done."
    )
    
    conversation_history = [
        TextMessage(content="Hi there!", source="user"),
        TextMessage(content="Hello! How can I help you today?", source="assistant"),
        TextMessage(content="I'm curious about the weather in London.", source="user"),
        TextMessage(content="The weather in London is 55°F.", source="assistant"),
        TextMessage(content="Can you check if it's comfortable?", source="user"),
    ]

    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break
        
        conversation_history.append(TextMessage(content=user_input, source="user"))

        # Fetch relevant memories
        relevant_memories = client.search(user_input, user_id="haris")
        print(relevant_memories)
        memory_context = "\n".join([mem['memory'] for mem in relevant_memories]) if relevant_memories else "No prior memory available."

        # Add memory context to the system message
        agent.system_message = f"You are a helpful assistant. Use the provided functions as needed. Relevant memory context: {memory_context}"

        # Process conversation
        response = await agent.on_messages(conversation_history, CancellationToken())
        
        # Append the assistant's reply
        conversation_history.append(response.chat_message)
        
        print("Assistant:", response.chat_message.content)

    # Store the conversation in memory
    msgs = [{"role": msg.source, "content": msg.content} for msg in conversation_history]
    client.add(msgs, user_id="haris")

asyncio.run(main())
