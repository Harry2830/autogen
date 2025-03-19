import os
from autogen import ConversableAgent
from typing import Annotated
from dotenv import load_dotenv
import requests

load_dotenv()

model = "gpt-3.5-turbo"
llm_config = {
    "model":model,
    "temperature":0.0,
    "api_key":os.environ["OPENAI_API_KEY"]
}



def get_flight_status(flight_number: Annotated[str, "Flight number"]) -> str:
    API_KEY = os.environ.get("AVIATIONSTACK_API_KEY")
    url = f"http://api.aviationstack.com/v1/flights?access_key={API_KEY}&flight_iata={flight_number}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        try:
            if data["data"] and data["data"][1]:
                flight_info = data["data"][1] 
                status = flight_info.get("flight_status", "Unknown")
                departure = flight_info.get("departure", "Unknown").get("airport","Unknown")
                arrival = flight_info.get("arrival", "Unknown").get("airport","Unknown")
            
                return f"The current status of flight {flight_number} is {status}. The Arrival airport of this flight is {arrival} and departure of this flight is {departure}."
            else:   
                return f"No data found for flight {flight_number}."
        except:
            
    else:
        return "Error fetching flight data."


def get_hotel_info(location:Annotated[str,"Location"])->str:
    dummy_data = {
        "New York":"Top Hotel in New York: The Plaza - 5 stars",
        "Los Angeles":"Top Hotel in New York: The Beverly Hills - 5 stars",
        "Chicago":"Top Hotel in Chicago: The Langham - 5 stars"
    }
    return dummy_data.get(location,f"No Hotels found in {location}")
    
def get_travel_advice(location:Annotated[str,"Location"])->str:
    dummy_data = {
        "New York":"Visit Times Square for the iconic city lights!",
        "Los Angeles":"Head to Santa Monica Pier for ocean views!",
        "Chicago":"Explore Millennium Park and see The Bean!"
    }
    return dummy_data.get(location,f"No Travel advice available")

assistant = ConversableAgent(name="TravelAssistant",
                            system_message="You are a helpful AI travel assistant. Return 'TERMINATE' when the task is done.",
                            llm_config=llm_config)

user_proxy = ConversableAgent(name="User",
                            is_termination_msg=lambda msg:msg.get("content") is not None and "TERMINATE" in msg["content"],
                            human_input_mode="ALWAYS")

assistant.register_for_llm(
    name="get_flight_status",
    description="Get the current status of the flight based on the flight number"
)(get_flight_status)
assistant.register_for_llm(
    name="get_hotel_info",
    description="Get information about hotels in a specific location"
)(get_hotel_info)
assistant.register_for_llm(
    name="get_travel_advice",
    description="Get Travel advice for a specific location"
)(get_travel_advice)
                
user_proxy.register_for_execution(name="get_flight_status")(get_flight_status)
user_proxy.register_for_execution(name="get_hotel_info")(get_hotel_info)
user_proxy.register_for_execution(name="get_travel_advice")(get_travel_advice)

user_proxy.initiate_chat(assistant, message="what is the current status of the flight EK516. also provide an travel advice to me for the New York.")