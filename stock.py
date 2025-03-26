import http.client
import json
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from dotenv import load_dotenv
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
import openai
import os
load_dotenv()

# Initialize environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RAPID_API_KEY = os.getenv("RAPID_API_KEY")
# Set OpenAI API Key
openai.api_key = OPENAI_API_KEY

# Initialize OpenAIChatCompletionClient
model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini",    
    api_key=OPENAI_API_KEY,
    temperature=0.7
)



# Fetch real-time stock price data from Yahoo Finance API via RapidAPI
def fetch_stock_data(stock_name: str) -> str:
    """
    Fetch real-time stock price trends for a given stock symbol using Yahoo Finance API.
    """
    conn = http.client.HTTPSConnection("yahoo-finance15.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': RAPID_API_KEY,
        'x-rapidapi-host': "yahoo-finance15.p.rapidapi.com"
    }

    # Request stock data for the given stock
    conn.request("GET", f"/api/v1/markets/stock/quotes?ticker={stock_name}", headers=headers)

    res = conn.getresponse()
    data = res.read()
    
    # Parse the response and extract relevant data
    stock_data = json.loads(data.decode("utf-8"))
    
    # Extracting a simple response message (you can adjust as needed)
    if stock_data.get("body"):
        quote = stock_data["body"][0]
        
        stock_symbol = quote.get("symbol", "Unknown")
        stock_price = quote.get("regularMarketPrice", "Unknown")
        market_change = quote.get("regularMarketChange", "Unknown")
        market_change_percent = quote.get("regularMarketChangePercent", "Unknown")
        day_high = quote.get("regularMarketDayHigh", "Unknown")
        day_low = quote.get("regularMarketDayLow", "Unknown")
        fifty_two_week_range = quote.get("fiftyTwoWeekRange", "Unknown")
        
        return (f"The current stock price of {stock_symbol} is ${stock_price}.\n"
                f"Market Change: ${market_change} ({market_change_percent}%)\n"
                f"Today's High: ${day_high} | Today's Low: ${day_low}\n"
                f"52-Week Range: {fifty_two_week_range}")
    else:
        return "Could not retrieve stock data. Please try again later."


# Fetch latest news related to a stock (you can modify or expand this functionality)
def fetch_stock_news(stock_name: str) -> str:
    """
    Fetch the latest news related to a given stock symbol using Yahoo Finance API via RapidAPI.
    """
    conn = http.client.HTTPSConnection("yahoo-finance15.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': RAPID_API_KEY,
        'x-rapidapi-host': "yahoo-finance15.p.rapidapi.com"
    }

    # Request news related to the given stock
    conn.request("GET", f"/api/v1/markets/news?ticker={stock_name}", headers=headers)

    res = conn.getresponse()
    data = res.read()
    
    # Parse the response and extract news data
    news_data = json.loads(data.decode("utf-8"))
    
    # Check for "body" in the response to ensure news items are available
    if "body" in news_data:
        news_articles = news_data["body"]
        news_summary = "Latest news:\n"
        
        for article in news_articles[:3]:  # Limiting to top 3 articles
            title = article.get("title", "No title")
            link = article.get("link", "No link")
            pub_date = article.get("pubDate", "No date")
            source = article.get("source", "No source")
            news_summary += f"- {title} ({link}) - Source: {source} - Published on: {pub_date}\n"
        
        return news_summary
    else:
        return "Could not fetch news. Please try again later."


# Agent functions using the Yahoo Finance API

async def stock_price_trends_tool(stock_name: str) -> str:
    """
    Fetch and return stock price trends for the stock using Yahoo Finance API.
    """
    print(f"[stock_price_trends_tool] Fetching stock price trends for {stock_name}...")
    return fetch_stock_data(stock_name)


async def news_analysis_tool(stock_name: str) -> str:
    """
    Fetch and return latest news for the stock using Yahoo Finance API.
    """
    print(f"[news_analysis_tool] Fetching news for {stock_name}...")
    return fetch_stock_news(stock_name)


# Agent function implementations remain the same

async def stock_price_trends_agent(stock_name: str) -> str:
    """Agent function for 'stock trends', calls stock_price_trends_tool."""
    return await stock_price_trends_tool(stock_name)

async def news_analysis_agent(stock_name: str) -> str:
    """Agent function for 'latest news', calls news_analysis_tool."""
    return await news_analysis_tool(stock_name)


###############################################################################
#                              AGENT DEFINITIONS
###############################################################################
stock_trends_agent_assistant = AssistantAgent(
    name="stock_trends_agent",
    model_client=model_client,
    tools=[stock_price_trends_agent],
    system_message=(
        "You are the Stock Price Trends Agent. "
        "You fetch and summarize stock prices, changes over the last few months, and general market trends. "
        "Do NOT provide any final investment decision."
    )
)

news_agent_assistant = AssistantAgent(
    name="news_agent",
    model_client=model_client,
    tools=[news_analysis_agent],
    system_message=(
        "You are the News Agent. "
        "You retrieve and summarize the latest news stories related to the given stock. "
        "Do NOT provide any final investment decision."
    )
)

sentiment_agent_assistant = AssistantAgent(
    name="sentiment_agent",
    model_client=model_client,
    tools=[],
    system_message=(
        "You are the Market Sentiment Agent. "
        "You gather overall market sentiment, relevant analyst reports, and expert opinions. "
        "Do NOT provide any final investment decision."
    )
)

decision_agent_assistant = AssistantAgent(
    name="decision_agent",
    model_client=model_client,
    system_message=(
        "You are the Decision Agent. After reviewing the stock data, news, sentiment, analyst reports, "
        "and expert opinions from the other agents, you provide the final investment decision. In the final decision make a call to either Invest or Not. Also provide the current stock price. "
        "End your response with 'Decision Made' once you finalize the decision."
    )
)

###############################################################################
#                        TERMINATION & TEAM CONFIGURATION
###############################################################################
text_termination = TextMentionTermination("Decision Made")
max_message_termination = MaxMessageTermination(15)
termination = text_termination | max_message_termination

investment_team = RoundRobinGroupChat(
    [
        stock_trends_agent_assistant,
        news_agent_assistant,
        sentiment_agent_assistant,
        decision_agent_assistant,
    ],
    termination_condition=termination
)

###############################################################################
#                                   MAIN
###############################################################################
async def main():
    # stock_name = "NVDA"
    stock_name = "AAPL"
    await Console(
        investment_team.run_stream(
            task=f"Analyze stock trends, news, and sentiment for {stock_name}, plus analyst reports and expert opinions, and then decide whether to invest."
        )
    )

if __name__ == "__main__":
    asyncio.run(main())
