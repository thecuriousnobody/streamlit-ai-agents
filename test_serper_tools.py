from search_tools_serper import serper_search_tool, serper_scholar_tool

def test_serper_search():
    print("\nTesting Serper Search Tool...")
    print("=" * 50)
    result = serper_search_tool.run("latest developments in artificial intelligence 2024")
    print(result)

def test_serper_scholar_search():
    print("\nTesting Serper Scholar Search Tool...")
    print("=" * 50)
    result = serper_scholar_tool.run("machine learning applications in healthcare")
    print(result)

if __name__ == "__main__":
    try:
        test_serper_search()
        test_serper_scholar_search()
    except Exception as e:
        print(f"Error during testing: {str(e)}")
