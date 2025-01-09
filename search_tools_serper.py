import os
import requests
from pydantic import Field
from typing import Optional
import streamlit as st

def serper_search(
    query: str = Field(description="The search query to execute")
) -> str:
    try:
        url = "https://google.serper.dev/search"
        payload = {
            "q": query
        }
        headers = {
            'X-API-KEY': st.secrets["SERPER_API_KEY"],
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
            
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"An error occurred while searching: {str(e)}"

def serper_scholar_search(
    query: str = Field(description="The academic search query to execute"),
    num_results: Optional[int] = Field(default=20, description="Number of results to return")
) -> str:
    try:
        # Modify the query to target scholarly content
        scholarly_query = f"{query} site:scholar.google.com"
        
        url = "https://google.serper.dev/search"
        payload = {
            "q": scholarly_query,
            "num": num_results
        }
        headers = {
            'X-API-KEY': st.secrets["SERPER_API_KEY"],
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Process and format the results
        formatted_results = []
        
        # Handle organic results
        for item in data.get('organic', []):
            title = item.get('title', 'No title')
            link = item.get('link', 'No link')
            snippet = item.get('snippet', 'No snippet')
            
            # Try to extract year and citations if available in the snippet
            year = "N/A"
            citations = "N/A"
            
            # Basic parsing of common Google Scholar patterns
            snippet_lower = snippet.lower()
            if '20' in snippet or '19' in snippet:  # Look for year
                for word in snippet.split():
                    if word.isdigit() and (1900 < int(word) < 2024):
                        year = word
                        break
                        
            if 'cited by' in snippet_lower:
                try:
                    cited_index = snippet_lower.index('cited by')
                    potential_number = snippet[cited_index:].split()[2]
                    if potential_number.isdigit():
                        citations = potential_number
                except:
                    pass
            
            formatted_results.append(f"Title: {title}")
            formatted_results.append(f"Link: {link}")
            formatted_results.append(f"Year: {year}")
            formatted_results.append(f"Citations: {citations}")
            formatted_results.append(f"Snippet: {snippet}")
            formatted_results.append("---")
        
        if not formatted_results:
            return "No scholarly results found or error in search."
            
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"An error occurred while searching scholar: {str(e)}"

# Create both tools using StructuredTool for consistency
from langchain.tools import StructuredTool

serper_search_tool = StructuredTool.from_function(
    func=serper_search,
    name="Internet Search",
    description="Search the internet for current information using Serper API."
)

serper_scholar_tool = StructuredTool.from_function(
    func=serper_scholar_search,
    name="Scholar Search",
    description="Search for academic papers and scholarly content using Serper API."
)
