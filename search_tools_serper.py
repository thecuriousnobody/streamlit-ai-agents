import os
import requests
from langchain.tools import Tool
from typing import Optional
import streamlit as st

def get_api_key(key_name: str) -> str:
    """Get API key from environment variables with fallback to Streamlit secrets"""
    # Try getting from environment first
    value = os.getenv(key_name)
    
    # If not in environment, try Streamlit secrets
    if not value and hasattr(st, 'secrets'):
        value = st.secrets.get(key_name)
        
    if not value:
        raise RuntimeError(f"Missing {key_name}. Set it in environment or secrets.toml")
        
    return value

def serper_search(query: str, result_as_answer: bool = True) -> str:
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": query
        }
        headers = {
            'X-API-KEY': get_api_key("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Process and format the results
        formatted_results = []
        
        # Handle organic results
        for item in data.get('organic', []):
            formatted_results.append(f"Title: {item.get('title', 'No title')}")
            formatted_results.append(f"Link: {item.get('link', 'No link')}")
            formatted_results.append(f"Snippet: {item.get('snippet', 'No snippet')}")
            formatted_results.append("---")
        
        if not formatted_results:
            return "No results found or error in search."
            
        result = "\n".join(formatted_results)
        return result if result_as_answer else result
        
    except Exception as e:
        error_msg = f"An error occurred while searching: {str(e)}"
        return error_msg if result_as_answer else error_msg

def serper_scholar_search(query: str, num_results: int = 20, result_as_answer: bool = True) -> str:
    try:
        url = "https://google.serper.dev/scholar"
        payload = {
            "q": query,
            "num": num_results
        }
        headers = {
            'X-API-KEY': get_api_key("SERPER_API_KEY"),
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Process and format the results
        formatted_results = []
        
        # Handle scholar results
        for item in data.get('organic', []):
            title = item.get('title', 'No title')
            link = item.get('link', 'No link')
            snippet = item.get('snippet', 'No snippet')
            publication = item.get('publication', 'N/A')
            year = item.get('year', 'N/A')
            citations = item.get('citations', 'N/A')
            
            formatted_results.append(f"Title: {title}")
            formatted_results.append(f"Publication: {publication}")
            formatted_results.append(f"Year: {year}")
            formatted_results.append(f"Citations: {citations}")
            formatted_results.append(f"Link: {link}")
            formatted_results.append(f"Summary: {snippet}")
            formatted_results.append("---")
        
        if not formatted_results:
            return "No scholarly results found or error in search."
            
        result = "\n".join(formatted_results)
        return result if result_as_answer else result
        
    except Exception as e:
        error_msg = f"An error occurred while searching scholar: {str(e)}"
        return error_msg if result_as_answer else error_msg

def search_wrapper(input_data):
    """Wrapper to handle different input formats with result_as_answer support"""
    result_as_answer = True  # Default to True for direct tool output
    
    if isinstance(input_data, str):
        return serper_search(input_data, result_as_answer=result_as_answer)
    if isinstance(input_data, dict):
        query = input_data.get('query', '')
        result_as_answer = input_data.get('result_as_answer', True)
        return serper_search(query, result_as_answer=result_as_answer)
    return serper_search(str(input_data), result_as_answer=result_as_answer)

def scholar_wrapper(input_data):
    """Wrapper to handle different input formats with result_as_answer support"""
    result_as_answer = True  # Default to True for direct tool output
    
    if isinstance(input_data, str):
        return serper_scholar_search(input_data, result_as_answer=result_as_answer)
    if isinstance(input_data, dict):
        query = input_data.get('query', '')
        num_results = input_data.get('num_results', 20)
        result_as_answer = input_data.get('result_as_answer', True)
        return serper_scholar_search(query, num_results, result_as_answer=result_as_answer)
    return serper_scholar_search(str(input_data), result_as_answer=result_as_answer)

# Create tools using Tool class with result_as_answer support
serper_search_tool = Tool(
    name="Internet Search",
    func=search_wrapper,
    description="Search the internet for current information using Serper API.",
    return_direct=True  # This ensures the tool output is returned directly
)

serper_scholar_tool = Tool(
    name="Scholar Search",
    func=scholar_wrapper,
    description="Search for academic papers and scholarly content using Serper API.",
    return_direct=True  # This ensures the tool output is returned directly
)
