import os
from crewai import Agent, Task, Crew, Process, LLM
from langchain.tools import Tool
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def search_api_search(query):
    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SEARCH_API_KEY")
        }
        response = requests.get(url, params=params)
        return response.json()
    except Exception as e:
        logger.error(f"Search API error: {str(e)}")
        return {"organic_results": []}

def run_search(query: str) -> str:
    results = search_api_search(query)
    
    # Process and format the results
    formatted_results = []
    for item in results.get('organic_results', []):
        formatted_results.append(f"Title: {item.get('title')}")
        formatted_results.append(f"Link: {item.get('link')}")
        formatted_results.append(f"Snippet: {item.get('snippet')}")
        formatted_results.append("---")
    
    return "\n".join(formatted_results)

search_tool = Tool(
    name="Internet Search",
    func=run_search,
    description="Useful for finding current data and information on various topics using the SearchAPI."
)

# Check for required environment variables
anthropic_key = os.getenv("ANTHROPIC_API_KEY")
search_key = os.getenv("SEARCH_API_KEY")

if not anthropic_key or not search_key:
    raise ValueError("Please set both ANTHROPIC_API_KEY and SEARCH_API_KEY environment variables")

ClaudeSonnet = LLM(
    api_key=anthropic_key,
    model="claude-3-5-sonnet-20241022",
) 

# Create a single agent
historical_analyst = Agent(
    role="Historical Analyst",
    goal="Analyze historical development of given topic",
    backstory="Expert in South Asian history specializing in cultural transformation",
    tools=[search_tool],
    llm=ClaudeSonnet
)

# Create a single task
historical_analysis = Task(
    description="Analyze historical development of South Asian cultural practices",
    agent=historical_analyst,
    expected_output="Detailed historical analysis with verified sources"
)

# Create crew with minimal configuration
crew = Crew(
    agents=[historical_analyst],
    tasks=[historical_analysis],
    verbose=True,
    process=Process.sequential
)

if __name__ == "__main__":
    # Run the crew
    result = crew.kickoff()
    print("\nResult:", result)
