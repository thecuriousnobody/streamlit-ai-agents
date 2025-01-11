import os
import requests
from langchain.tools import Tool
from pydantic import BaseModel, Field
# from pydantic.v1 import BaseModel, Field
from typing import Optional
import streamlit as st

class SerperSearchSchema(BaseModel):
    query: str = Field(description="The search query to execute")

class SerperScholarSearchSchema(BaseModel):
    query: str = Field(description="The academic search query to execute")
    num_results: Optional[int] = Field(default=20, description="Number of results to return")

def serper_search(query: str) -> str:
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

def serper_scholar_search(query: str, num_results: int = 20) -> str:
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
            
            # Extract year and citations from the title and snippet
            year = "N/A"
            citations = "N/A"
            
            # Look for year in title and snippet
            for text in [title, snippet]:
                if not year or year == "N/A":
                    words = text.split()
                    for word in words:
                        # Clean the word from any punctuation
                        clean_word = ''.join(c for c in word if c.isdigit())
                        if clean_word.isdigit() and 1900 < int(clean_word) < 2025:
                            year = clean_word
                            break
            
            # Look for citations in snippet
            snippet_lower = snippet.lower()
            citation_patterns = ['cited by', 'citations:', 'citations -']
            for pattern in citation_patterns:
                if pattern in snippet_lower:
                    try:
                        pattern_index = snippet_lower.index(pattern)
                        # Look at the next few words for a number
                        following_text = snippet[pattern_index:pattern_index + 30].split()
                        for word in following_text:
                            clean_word = ''.join(c for c in word if c.isdigit())
                            if clean_word.isdigit():
                                citations = clean_word
                                break
                    except:
                        continue
            
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

# Create tools using Tool class for consistency with search_tools.py
serper_search_tool = Tool(
    name="Internet Search",
    func=serper_search,
    description="Search the internet for current information using Serper API.",
    args_schema=SerperSearchSchema
)

serper_scholar_tool = Tool(
    name="Scholar Search",
    func=serper_scholar_search,
    description="Search for academic papers and scholarly content using Serper API.",
    args_schema=SerperScholarSearchSchema
)
