import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from search_tools import search_tool, youtube_tool, search_api_tool

# Set page config
st.set_page_config(
    page_title="Podcast Guest Finder",
    page_icon="🎙️",
    layout="wide"
)

# Initialize environment variables
os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY
os.environ["ANTHROPIC_API_KEY"] = config.ANTHROPIC_API_KEY
os.environ["SERPAPI_API_KEY"] = config.SERPAPI_API_KEY

# Initialize SerpAPI
search_wrapper = SerpAPIWrapper(
    serpapi_api_key=config.SERPAPI_API_KEY,
    params={
        "engine": "google",
        "google_domain": "google.com",
        "gl": "us",
        "hl": "en",
        "num": 10,
        "tbm": "",
        "safe": "active",
        "device": "desktop",
        "output": "json",
        "no_cache": False
    }
)

ClaudeSonnet = LLM(
    api_key=config.ANTHROPIC_API_KEY,
    model="claude-3-5-sonnet-20241022",
)

# Create a custom tool using the SerpAPIWrapper
search_tool = Tool(
    name="Search",
    func=search_wrapper.run,
    description="Useful for searching the internet to find information on people, topics, or current events."
)

def create_agents_and_tasks(niche_topic):
    topic_analyzer = Agent(
        role="Topic Analyzer",
        goal=f"Analyze the provided niche topic and identify key aspects, related fields, and potential perspectives for discussion.",
        backstory="You are an expert at breaking down complex topics and identifying various angles for in-depth discussion.",
        verbose=True,
        allow_delegation=False,
        llm=ClaudeSonnet
    )

    expert_finder = Agent(
        role="Expert Finder",
        goal=f"Identify a diverse range of potential guests related to the analyzed topic.",
        backstory="You have vast knowledge of people working across various disciplines, with a focus on both well-known experts and lesser-known but impactful individuals.",
        verbose=True,
        allow_delegation=False,
        llm=ClaudeSonnet,
        tools=[search_api_tool]
    )

    contact_researcher = Agent(
        role="Contact Information Researcher",
        goal=f"Find contact information for the identified potential guests.",
        backstory="You are skilled at finding contact information for individuals across various sectors.",
        verbose=True,
        allow_delegation=False,
        llm=ClaudeSonnet,
        tools=[search_api_tool]
    )

    analyze_topic_task = Task(
        description=f"Analyze the following niche topic in depth: '{niche_topic}'. Identify key aspects, related fields, historical context, current relevance, and potential perspectives for discussion.",
        agent=topic_analyzer,
        expected_output="A detailed analysis of the niche topic, including key aspects, related fields, historical context, current relevance, and potential perspectives for discussion."
    )

    find_experts_task = Task(
        description=f"Based on the analysis of the niche topic '{niche_topic}', identify a diverse range of potential guests who could provide valuable insights as podcast guests. Include individuals with high, medium, and low public profiles.",
        agent=expert_finder,
        expected_output="""A comprehensive list of at least 15 potential podcast guests, including:
        1. Their names and roles/affiliations
        2. A brief description of their work or experience related to the topic
        3. Why they would be a valuable guest (their unique perspective or contribution)
        4. Their approximate level of public profile (high, medium, low)
        
        Ensure the list includes a balanced mix of high-profile experts, mid-level professionals, and lesser-known individuals doing important work in the field. Aim for at least 5 individuals in each category (high, medium, low profile).""",
        context=[analyze_topic_task]
    )

    research_contacts_task = Task(
        description=f"For the identified potential guests, research and provide their contact information or suggest ways to reach them.",
        agent=contact_researcher,
        expected_output="""For each potential guest:
        1. Any available contact information
        2. Suggestions for reaching out if direct contact info is not available
        3. Notes on the best approach for contacting each individual (e.g., through their organization, via social media, etc.)
        4. Any relevant etiquette or cultural considerations for reaching out to these individuals""",
        context=[find_experts_task]
    )

    return [topic_analyzer, expert_finder, contact_researcher], [analyze_topic_task, find_experts_task, research_contacts_task]

def run_guest_finder(niche_topic, progress_containers):
    agents, tasks = create_agents_and_tasks(niche_topic)
    
    # Initialize all containers with waiting state
    progress_containers["topic"].info("⏳ Waiting to start topic analysis...")
    progress_containers["experts"].info("⏳ Waiting to start expert search...")
    progress_containers["contacts"].info("⏳ Waiting to start contact research...")
    
    # Start Topic Analysis
    with progress_containers["topic"].status("🔄 Analyzing topic..."):
        progress_containers["topic"].write("This may take a few minutes...")
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )

    def process_output(output):
        # Split the output based on agent sections
        sections = output.split("# Agent:")
        
        for section in sections[1:]:  # Skip the first empty section
            if "Topic Analyzer" in section:
                progress_containers["topic"].success("✅ Topic Analysis Complete")
                progress_containers["topic"].markdown(section.split("# Agent:")[0].strip())
                # Start Expert Search
                progress_containers["experts"].status("🔄 Searching for experts...")
                progress_containers["experts"].write("This may take a few minutes...")
            elif "Expert Finder" in section:
                progress_containers["experts"].success("✅ Expert Search Complete")
                progress_containers["experts"].markdown(section.split("# Agent:")[0].strip())
                # Start Contact Research
                progress_containers["contacts"].status("🔄 Researching contact information...")
                progress_containers["contacts"].write("This may take a few minutes...")
            elif "Contact Information Researcher" in section:
                progress_containers["contacts"].success("✅ Contact Research Complete")
                progress_containers["contacts"].markdown(section.split("# Agent:")[0].strip())
        
        return output

    output = crew.kickoff()
    # Convert CrewOutput to string before processing
    output_str = str(output)
    return process_output(output_str)

# Streamlit UI
st.title("🎙️ Podcast Guest Finder")
st.write("Find the perfect guests for your podcast based on your niche topic!")

# Input section
with st.form("guest_finder_form"):
    niche_topic = st.text_input("Enter your podcast's niche topic:", 
                               placeholder="e.g., Sustainable Urban Agriculture")
    submitted = st.form_submit_button("Find Guests")

# Process and display results
if submitted and niche_topic:
    # Create containers for each step
    progress_containers = {
        "topic": st.container(),
        "experts": st.container(),
        "contacts": st.container()
    }
    
    # Add headers for each section
    with progress_containers["topic"]:
        st.subheader("1️⃣ Topic Analysis")
    with progress_containers["experts"]:
        st.subheader("2️⃣ Expert Search")
    with progress_containers["contacts"]:
        st.subheader("3️⃣ Contact Research")
    
    try:
        result = run_guest_finder(niche_topic, progress_containers)
        
        # Create download button
        st.download_button(
            label="Download Results",
            data=result,
            file_name=f"potential_guests_{niche_topic.replace(' ', '_')}.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please try again with a different topic or contact support if the issue persists.")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use:
1. Enter your podcast's niche topic in the text field
2. Click 'Find Guests' to start the analysis
3. Wait for the AI to analyze your topic and find potential guests
4. Download the results for future reference

The AI will provide:
- Topic analysis
- Potential guest recommendations
- Contact information and outreach strategies
""")
