import os
import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
from autogen import ConversableAgent

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Headers for web requests
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# === Define Agents ===
manager = ConversableAgent(
    name="Manager",
    system_message="You oversee the web scraping process and assign tasks to other agents.",
    llm_config={"model": "gpt-4o-mini", "api_key": API_KEY}
)

scraper_agent = ConversableAgent(
    name="Scraper",
    system_message="You scrape web pages and extract content.",
    llm_config={"model": "gpt-4o-mini", "api_key": API_KEY}
)

processor_agent = ConversableAgent(
    name="Processor",
    system_message="You process and clean scraped web content, removing irrelevant data.",
    llm_config={"model": "gpt-4o-mini", "api_key": API_KEY}
)

answer_agent = ConversableAgent(
    name="AnswerBot",
    system_message="You analyze processed web content and answer user queries.",
    llm_config={"model": "gpt-4o-mini", "api_key": API_KEY}
)

# === Functions ===
def scrape_page(url: str) -> str:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        for script in soup(["script", "style", "noscript"]):
            script.extract()
        text_content = soup.get_text(separator="\n", strip=True)
        return text_content[:8000] if text_content else f"No meaningful text found on {url}."
    except requests.exceptions.RequestException as e:
        return f"Error fetching {url}: {e}"

def find_internal_links(url: str) -> set:
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        base_domain = urlparse(url).netloc
        links = {urljoin(url, link["href"]) for link in soup.find_all("a", href=True)}
        return {link for link in links if urlparse(link).netloc == base_domain}
    except requests.exceptions.RequestException:
        return set()

def scrape_website(url: str, max_pages=5) -> dict:
    visited, pages_data, to_visit = set(), {}, {url}
    count = 0
    while to_visit and count < max_pages:
        page_url = to_visit.pop()
        if page_url in visited:
            continue
        text = scrape_page(page_url)
        pages_data[page_url] = text
        visited.add(page_url)
        count += 1
        to_visit.update(find_internal_links(page_url) - visited)
        time.sleep(1)
    return pages_data

def process_scraped_content(scraped_data: dict) -> str:
    return "\n\n".join(scraped_data.values())[:8000]

def answer_from_scraped_content(content: str, question: str) -> str:
    response = answer_agent.generate_reply(
        messages=[{"role": "user", "content": f"Content:\n\n{content}\n\nQuestion: {question}"}]
    )
    return response["content"]

# === Register Agent Functions ===
scraper_agent.register_for_llm(name="scrape_website", description="Scrapes content from a website.")(scrape_website)
processor_agent.register_for_llm(name="process_content", description="Processes web content.")(process_scraped_content)
answer_agent.register_for_llm(name="answer_question", description="Answers queries about content.")(answer_from_scraped_content)

def interactive_chat():
    print("\n🤖 Multi-Agent Web Scraper Initialized! Type 'exit' anytime.\n")
    while True:
        url = input("🌐 Enter a website URL to scrape: ").strip()
        if url.lower() == "exit":
            print("👋 Exiting!")
            break
        max_pages = int(input("🔢 Max pages to scrape (default 5): ") or 5)
        import json

        scraped_data = scraper_agent.initiate_chat(
    manager, 
    message={"content": "Scrape the website.", "function_call": {"name": "scrape_website", "arguments": json.dumps({"url": url, "max_pages": max_pages})}}
)


        processed_data = processor_agent.initiate_chat(
    manager, 
    message={"content": "Process the scraped content.", "function_call": {"name": "process_content", "arguments": {"scraped_data": scraped_data}}}
)

        while True:
            question = input("❓ Ask a question or type 'new'/'exit': ").strip()
            if question.lower() == "exit":
                return
            elif question.lower() == "new":
                break
            answer = answer_agent.initiate_chat(
    manager, 
    message={"content": "Answer the user's question.", "function_call": {"name": "answer_question", "arguments": {"content": processed_data, "question": question}}}
)

            print(f"💡 Answer: {answer}\n")

if __name__ == "__main__":
    interactive_chat()
