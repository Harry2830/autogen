import asyncio
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

def get_weather(city: str) -> str:
    """Fetch current temperature for a given city."""
    if city.lower() == "new york":
        return "75°F"
    elif city.lower() == "los angeles":
        return "85°F"
    elif city.lower() == "london":
        return "55°F"
    else:
        return "Unknown city"

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
    
    # Create the assistant with both tools and reflection enabled.
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[get_weather, check_weather],
        reflect_on_tool_use=True,
        system_message="You are a helpful assistant. Use the provided functions as needed to answer the question. Reply with TERMINATE when done."
    )
    
    # Pre-populate conversation history with five messages.
    conversation_history = [
        TextMessage(content="Hi there!", source="user"),
        TextMessage(content="Hello! How can I help you today?", source="assistant"),
        TextMessage(content="I'm curious about the weather in London.", source="user"),
        TextMessage(content="The weather in London is 55°F.", source="assistant"),
        TextMessage(content="Can you check if it's comfortable?", source="user"),
    ]
    
    # Optionally display the initial conversation context.
    # print("Pre-populated conversation context:")
    # for msg in conversation_history:
    #     print(f"{msg.source.capitalize()}: {msg.content}")
    # print("-----")
    
    # Interactive loop: each new message is appended and the full conversation is passed.
    while True:
        user_input = input("User: ")
        if user_input.strip().lower() == "exit":
            break
        
        # Add the new user message to the conversation history.
        conversation_history.append(TextMessage(content=user_input, source="user"))
        
        # The assistant processes the full conversation history.
        response = await agent.on_messages(conversation_history, CancellationToken())
        # Append the assistant's reply to the conversation history.
        conversation_history.append(response.chat_message)
        
        print("Assistant:", response.chat_message.content)

asyncio.run(main())
