import os
import requests
from langchain.tools import Tool
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional

# Load environment variables
load_dotenv()

class SearchSchema(BaseModel):
    query: str = Field(description="The search query to execute")

class ScholarSearchSchema(BaseModel):
    query: str = Field(description="The academic search query to execute")
    num_results: Optional[int] = Field(default=20, description="Number of results to return")

class NewsArchiveSchema(BaseModel):
    query: str = Field(description="The news search query to execute")
    start_year: Optional[int] = Field(default=None, description="Start year for date range")
    end_year: Optional[int] = Field(default=None, description="End year for date range")

class LocalArchivesSchema(BaseModel):
    query: str = Field(description="The archive search query to execute")
    language: Optional[str] = Field(default=None, description="Language for the search")

class LegalDatabaseSchema(BaseModel):
    query: str = Field(description="The legal search query to execute")
    document_type: Optional[str] = Field(default=None, description="Type of legal document to search for")
    date_range: Optional[str] = Field(default=None, description="Date range for the search")

class GovernmentArchivesSchema(BaseModel):
    query: str = Field(description="The government archive search query to execute")
    document_type: Optional[str] = Field(default=None, description="Type of government document to search for")

def search_api_search(query):
    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SEARCH_API_KEY")
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        data = response.json()
        if not data.get('organic_results'):
            return {
                "organic_results": [{
                    "title": "Limited or no results available",
                    "link": "",
                    "snippet": "The search returned no results. This could be due to rate limiting or the specificity of the query. Consider broadening your search terms or trying again later."
                }]
            }
            
        return data
    except requests.exceptions.RequestException as e:
        return {
            "organic_results": [{
                "title": "Search API Error",
                "link": "",
                "snippet": f"An error occurred while searching: {str(e)}. This might be temporary - please try again."
            }]
        }
    except Exception as e:
        return {
            "organic_results": [{
                "title": "Search Processing Error",
                "link": "",
                "snippet": "An unexpected error occurred while processing the search results. Please try again or modify your search terms."
            }]
        }

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
    description="Useful for finding current data and information on various topics using the SearchAPI.",
    args_schema=SearchSchema
)

def google_scholar_search(query, num_results=20):
    try:
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": os.getenv("SEARCH_API_KEY"),
            "num": num_results
        }
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
        return []

google_scholar_tool = Tool(
    name="Google Scholar Search",
    func=google_scholar_search,
    description="Academic search tool returning scholarly articles with citation metrics.",
    args_schema=ScholarSearchSchema
)

def news_archive_search(query, start_year=None, end_year=None):
    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": os.getenv("SEARCH_API_KEY"),
            "tbm": "nws",  # News search
            "tbs": "ar:1"  # Archive results
        }
        
        # Add date range if specified
        if start_year and end_year:
            params["tbs"] += f",cdr:1,cd_min:1/1/{start_year},cd_max:12/31/{end_year}"
        
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
        return []

news_archive_tool = Tool(
    name="News Archive Search",
    func=news_archive_search,
    description="Search historical news archives for relevant articles and coverage, with optional date range filtering.",
    args_schema=NewsArchiveSchema
)

def local_archives_search(query, language=None):
    try:
        # Construct a specialized query for local archives
        base_query = query
        if language:
            # Add language-specific keywords
            lang_map = {
                "assamese": "অসমীয়া",  # Assamese script
                "bengali": "বাংলা"      # Bengali script
            }
            lang_term = lang_map.get(language.lower(), language)
            base_query = f"{base_query} {lang_term} source:archive.org OR source:assamarchives OR source:statearchives"
        
        # Add specific terms to target archive content
        archive_query = (f"{base_query} site:archive.org OR site:assamarchives.gov.in "
                        f"filetype:pdf OR filetype:doc OR filetype:txt")
        
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google",
            "q": archive_query,
            "api_key": os.getenv("SEARCH_API_KEY"),
            "num": 20  # Increase results to get more relevant matches
        }
        
        response = requests.get(url, params=params)
        results = response.json()
        
        # Process and filter results
        parsed_results = []
        for item in results.get("organic_results", []):
            if any(term in item.get("link", "").lower() for term in ["archive", "repository", "historical", "manuscript"]):
                parsed_result = {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "source_type": "local_archive"
                }
                parsed_results.append(parsed_result)
        
        return parsed_results
    except Exception as e:
        return []

local_archives_tool = Tool(
    name="Local Archives Search",
    func=local_archives_search,
    description="Search local archives and repositories for historical documents and records.",
    args_schema=LocalArchivesSchema
)

def legal_database_search(query, document_type=None, date_range=None):
    try:
        # Construct a specialized legal query
        base_query = query
        
        # Add document type specific terms
        doc_type_terms = {
            "nrc": "National Register of Citizens NRC Assam",
            "citizenship": "Citizenship Amendment Act CAA Assam",
            "court_orders": "High Court Supreme Court order judgment Assam",
            "legislation": "legislation act law gazette notification Assam"
        }
        
        if document_type:
            base_query = f"{base_query} {doc_type_terms.get(document_type, '')}"
            
        # Add date range if specified
        if date_range:
            base_query = f"{base_query} {date_range}"
            
        # Add legal-specific search terms
        legal_query = (f"{base_query} site:http://www.asianlii.org/ "
                      f"filetype:pdf OR filetype:doc")
        
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google",
            "q": legal_query,
            "api_key": os.getenv("SEARCH_API_KEY"),
            "num": 20
        }
        
        response = requests.get(url, params=params)
        results = response.json()
        
        # Process and filter results
        parsed_results = []
        for item in results.get("organic_results", []):
            if any(term in item.get("link", "").lower() for term in ["court", "legal", "law", "gazette", "judgment"]):
                parsed_result = {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "doc_type": document_type or "general"
                }
                parsed_results.append(parsed_result)
        
        return parsed_results
    except Exception as e:
        return []

legal_database_tool = Tool(
    name="Legal Database Search",
    func=legal_database_search,
    description="Search legal databases for court decisions, legislation, and policy documents.",
    args_schema=LegalDatabaseSchema
)

def government_archives_search(query, document_type=None):
    try:
        # Construct a specialized government archives query
        base_query = query
        
        # Add document type specific terms
        doc_type_terms = {
            "policy": "policy implementation report notification",
            "statistics": "census demographic statistics data",
            "administrative": "administrative order circular memorandum",
            "development": "development project scheme program"
        }
        
        if document_type:
            base_query = f"{base_query} {doc_type_terms.get(document_type, '')}"
            
        # Add government-specific search terms
        gov_query = (f"{base_query} site:gov.in "
                    f"filetype:pdf OR filetype:doc")
        
        url = "https://www.searchapi.io/api/v1/search"
        params = {
            "engine": "google",
            "q": gov_query,
            "api_key": os.getenv("SEARCH_API_KEY"),
            "num": 20
        }
        
        response = requests.get(url, params=params)
        results = response.json()
        
        # Process and filter results
        parsed_results = []
        for item in results.get("organic_results", []):
            if any(term in item.get("link", "").lower() for term in ["gov", "government", "ministry", "department"]):
                parsed_result = {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                    "doc_type": document_type or "general"
                }
                parsed_results.append(parsed_result)
        
        return parsed_results
    except Exception as e:
        return []

government_archives_tool = Tool(
    name="Government Archives Search",
    func=government_archives_search,
    description="Search government archives for official documents, reports, and records.",
    args_schema=GovernmentArchivesSchema
)
