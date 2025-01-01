import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import logging
import requests
from langchain.tools import Tool

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def search_api_search(query):
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "google",
        "q": query,
        "api_key": st.secrets["SEARCH_API_KEY"]
    }
    response = requests.get(url, params=params)
    return response.json()

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

search_api_tool = Tool(
    name="Internet Search",
    func=run_search,
    description="Useful for finding current data and information on various topics using the SearchAPI."
)

def google_scholar_search(query, num_results=20):
    url = "https://www.searchapi.io/api/v1/search"
    params = {
        "engine": "google_scholar",
        "q": query,
        "api_key": st.secrets["SEARCH_API_KEY"],
        "num": num_results
    }
    
    try:
        response = requests.get(url, params=params)
        results = response.json()
        
        parsed_results = []
        for result in results.get("organic_results", []):
            parsed_result = {
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet"),
                "citations": result.get("cited_by_count"),
                "year": result.get("year"),
                "authors": result.get("authors", [])
            }
            parsed_results.append(parsed_result)
            
        return parsed_results
        
    except Exception as e:
        logger.error(f"Google Scholar search error: {str(e)}")
        return []

google_scholar_tool = Tool(
    name="Google Scholar Search",
    func=google_scholar_search,
    description="Academic search tool returning scholarly articles with citation metrics."
)

def news_archive_search(query, start_year=None, end_year=None):
    params = {
        "engine": "google",
        "q": query,
        "api_key": st.secrets["SEARCH_API_KEY"],
        "tbm": "nws",  # News search
        "tbs": "ar:1"  # Archive results
    }
    
    # Add date range if specified
    if start_year and end_year:
        params["tbs"] += f",cdr:1,cd_min:1/1/{start_year},cd_max:12/31/{end_year}"
    
    try:
        url = "https://www.searchapi.io/api/v1/search"
        response = requests.get(url, params=params)
        results = response.json()
        
        parsed_results = []
        for result in results.get("organic_results", []):
            parsed_result = {
                "title": result.get("title"),
                "link": result.get("link"),
                "snippet": result.get("snippet"),
                "source": result.get("source"),
                "date": result.get("date")
            }
            parsed_results.append(parsed_result)
            
        return parsed_results
        
    except Exception as e:
        logger.error(f"News archive search error: {str(e)}")
        return []

news_archive_tool = Tool(
    name="News Archive Search",
    func=news_archive_search,
    description="Search historical news archives for relevant articles and coverage, with optional date range filtering."
)
