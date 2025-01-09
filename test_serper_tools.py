import os
import traceback
from search_tools_serper import serper_search_tool, serper_scholar_tool

# Set up logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock streamlit secrets for testing
class MockSecrets:
    def __init__(self):
        self.secrets = {
            "SERPER_API_KEY": "0a49b8ac6532a531cca6f928e1c5a04bc580e547"
        }
    
    def __getitem__(self, key):
        logger.info(f"Accessing secret key: {key}")
        return self.secrets[key]

# Replace streamlit secrets with mock for testing
import sys
import streamlit as st
sys.modules['streamlit'].secrets = MockSecrets()

def test_search():
    logger.info("Starting basic search test...")
    try:
        query = "artificial intelligence latest developments"
        logger.info(f"Executing search query: {query}")
        results = serper_search_tool.run(query)
        logger.info("Search completed successfully")
        print(f"\nSearch Results for '{query}':")
        print(results)
    except Exception as e:
        logger.error(f"Error in search test: {str(e)}")
        logger.error(traceback.format_exc())

def test_scholar():
    logger.info("Starting scholar search test...")
    try:
        query = "machine learning neural networks"
        logger.info(f"Executing scholar query: {query}")
        results = serper_scholar_tool.invoke({"query": query, "num_results": 5})
        logger.info("Scholar search completed successfully")
        print(f"\nScholar Results for '{query}':")
        print(results)
    except Exception as e:
        logger.error(f"Error in scholar test: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    logger.info("Starting tests...")
    try:
        test_search()
        test_scholar()
        logger.info("All tests completed")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.error(traceback.format_exc())
