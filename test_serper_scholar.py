import os
from search_tools_serper import serper_scholar_search

def test_scholar_search():
    # Test query
    query = "machine learning neural networks"
    print(f"Testing scholar search with query: {query}")
    
    try:
        results = serper_scholar_search(query)
        print("\nResults:")
        print(results)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_scholar_search()
