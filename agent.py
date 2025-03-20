import os
from autogen import ConversableAgent
from typing import Annotated
from dotenv import load_dotenv
import requests
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

load_dotenv()

# Initialize memory manually
user_memory = ListMemory()

# OpenAI API Config
model = "gpt-3.5-turbo"
llm_config = {
    "model": model,
    "temperature": 0.0,
    "api_key": os.environ["OPENAI_API_KEY"]
}

# Function to get flight status
def get_flight_status(flight_number: Annotated[str, "Flight number"]) -> str:
    API_KEY = os.environ.get("AVIATIONSTACK_API_KEY")
    url = f"http://api.aviationstack.com/v1/flights?access_key={API_KEY}&flight_iata={flight_number}"
    
    response = requests.get(url)
    if response.status_code == 200: 
        data = response.json()
        try:
            if data["data"] and len(data["data"]) > 1:
                flight_info = data["data"][1]
                status = flight_info.get("flight_status", "Unknown")
                departure = flight_info.get("departure", {}).get("airport", "Unknown")
                arrival = flight_info.get("arrival", {}).get("airport", "Unknown")
                return f"The current status of flight {flight_number} is {status}. It departs from {departure} and arrives at {arrival}."
            else:
                return f"No data found for flight {flight_number}."
        except Exception as e:
            return f"Error processing flight data: {e}"
    else:
        return "Error fetching flight data."

# Function to get hotel info
def get_hotel_info(location: Annotated[str, "Location"]) -> str:
    dummy_data = {
        "New York": "Top Hotel in New York: The Plaza - 5 stars",
        "Los Angeles": "Top Hotel in Los Angeles: The Beverly Hills - 5 stars",
        "Chicago": "Top Hotel in Chicago: The Langham - 5 stars"
    }
    return dummy_data.get(location, f"No Hotels found in {location}")

# Function to get travel advice
def get_travel_advice(location: Annotated[str, "Location"]) -> str:
    dummy_data = {
        "New York": "Visit Times Square for the iconic city lights!",
        "Los Angeles": "Head to Santa Monica Pier for ocean views!",
        "Chicago": "Explore Millennium Park and see The Bean!"
    }
    return dummy_data.get(location, f"No Travel advice available")

# Create assistant agent
assistant = ConversableAgent(
    name="TravelAssistant",
    system_message="You are a helpful AI travel assistant. Return 'TERMINATE' when the task is done.",
    llm_config=llm_config
)

# Create user proxy agent
user_proxy = ConversableAgent(
    name="User",
    is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"],
    human_input_mode="TERMINATE"
)

# Update context function to use memory
def update_context(agent, context):
    """Update conversation context with memory."""
    memory_texts = [mem.content for mem in user_memory.get_all()]
    context["memory"] = "\n".join(memory_texts) if memory_texts else "No prior memory available."
    return context

# Attach the update_context function to assistant
assistant.update_context = update_context

# Register functions for LLM execution
assistant.register_for_llm(name="get_flight_status", description="Get the current status of the flight")(get_flight_status)
assistant.register_for_llm(name="get_hotel_info", description="Get information about hotels in a location")(get_hotel_info)
assistant.register_for_llm(name="get_travel_advice", description="Get travel advice for a location")(get_travel_advice)

# Register functions for execution
user_proxy.register_for_execution(name="get_flight_status")(get_flight_status)
user_proxy.register_for_execution(name="get_hotel_info")(get_hotel_info)
user_proxy.register_for_execution(name="get_travel_advice")(get_travel_advice)

# Store user preference in memory (Example)
user_memory.add(MemoryContent(content="User prefers travel-related information and flight status updates.", mime_type=MemoryMimeType.TEXT))

# Start conversation
user_proxy.initiate_chat(assistant, message="What is the current status of flight EK516? Also, provide travel advice for New York.")