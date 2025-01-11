import os
from unittest.mock import MagicMock
import streamlit as st
import sys

# Mock streamlit
st.text_input = MagicMock(return_value="Cultural transformation in Assam")
st.button = MagicMock(return_value=True)
st.session_state = {}
st.error = MagicMock()
st.write = MagicMock()
st.text_area = MagicMock()
st.success = MagicMock()

# Import the research module
from southAsianHistoryResearch_Render import create_agents_and_tasks, start_research

def test_research_flow():
    print("\nTesting South Asian History Research Flow...")
    print("=" * 50)
    
    # Test topic
    research_topic = "Cultural transformation in Assam"
    
    print(f"\nCreating agents and tasks for topic: {research_topic}")
    try:
        agents, tasks = create_agents_and_tasks(research_topic)
        if agents and tasks:
            print("✓ Successfully created agents and tasks")
            print(f"Number of agents: {len(agents)}")
            print(f"Number of tasks: {len(tasks)}")
            
            # Print agent roles
            print("\nAgent Roles:")
            for agent in agents:
                print(f"- {agent.role}")
            
            # Print task descriptions
            print("\nTasks:")
            for task in tasks:
                print(f"- Task for {task.agent.role}")
        else:
            print("✗ Failed to create agents and tasks")
            return
            
        print("\nTesting research execution...")
        start_research(research_topic)
        
        if st.session_state.get('research_results'):
            print("✓ Research completed successfully")
            print("\nResearch Results Preview (first 200 chars):")
            print(st.session_state['research_results'][:200] + "...")
        else:
            print("✗ No research results generated")
            
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")

if __name__ == "__main__":
    test_research_flow()
