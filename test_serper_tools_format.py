from search_tools_serper import serper_search_tool, serper_scholar_tool

def test_search_tool_format():
    print("\nTesting Internet Search with agent-style input format...")
    try:
        # Test with the exact format the agent uses
        result = serper_search_tool.run("latest developments in artificial intelligence")
        print("✅ Search successful!")
        print("\nSample results:")
        print(result[:500] + "..." if len(result) > 500 else result)
    except Exception as e:
        print(f"❌ Search failed with error: {str(e)}")

def test_scholar_tool_format():
    print("\nTesting Scholar Search with agent-style input format...")
    try:
        # Test with the exact format the agent uses
        result = serper_scholar_tool.run("machine learning neural networks")
        print("✅ Scholar search successful!")
        print("\nSample results:")
        print(result[:500] + "..." if len(result) > 500 else result)
    except Exception as e:
        print(f"❌ Scholar search failed with error: {str(e)}")

def test_scholar_tool_without_num_results():
    print("\nTesting Scholar Search with minimal input format...")
    try:
        # Test with just the required parameter
        result = serper_scholar_tool.run("artificial intelligence ethics")
        print("✅ Scholar search (without num_results) successful!")
        print("\nSample results:")
        print(result[:500] + "..." if len(result) > 500 else result)
    except Exception as e:
        print(f"❌ Scholar search failed with error: {str(e)}")

if __name__ == "__main__":
    print("Testing Serper tools with agent-style input format...")
    print("=" * 80)
    
    test_search_tool_format()
    print("\n" + "=" * 80)
    
    test_scholar_tool_format()
    print("\n" + "=" * 80)
    
    test_scholar_tool_without_num_results()
    print("\n" + "=" * 80)
    
    print("\nTests completed!")
