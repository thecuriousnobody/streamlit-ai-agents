import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
from mem0 import MemoryClient
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_tools_docker import search_api_tool, google_scholar_tool, news_archive_tool
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pysqlite3 as sqlite3

# Set page config
st.set_page_config(
    page_title="South Asian History Research",
    page_icon="📚",
    layout="wide"
)

# Initialize LLM and Mem0 with API keys from environment variables
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
mem0_api_key = os.getenv("MEM0_API_KEY")

if not anthropic_api_key or not mem0_api_key:
    st.error("""
        Please set up your API keys as environment variables:
        
        Required environment variables:
        - ANTHROPIC_API_KEY
        - MEM0_API_KEY
        - SEARCH_API_KEY
        
        These should be provided when running the Docker container.
    """)
    st.stop()

from langchain_anthropic import ChatAnthropic

# Initialize the Anthropic chat model
base_model = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    max_tokens=8192,
    temperature=0.6,
    anthropic_api_key=anthropic_api_key
)

# Wrap it in CrewAI's LLM
ClaudeSonnet = LLM(base_llm=base_model)

# Initialize Mem0 client
mem0_client = MemoryClient(api_key=mem0_api_key)

def create_agents_and_tasks(research_topic):
    historical_analyst = Agent(
        role="Historical Analyst",
        goal="Analyze historical development of Assamese Hindu culture and Muslim communities",
        backstory="Expert in South Asian history specializing in cultural transformation and minority experiences",
        tools=[google_scholar_tool, search_api_tool],
        llm=ClaudeSonnet
    )

    media_analyzer = Agent(
        role="Media Content Analyzer",
        goal="Track and analyze media representations of Muslim communities in Assam",
        backstory="Media studies expert focusing on minority representation and narrative analysis",
        tools=[search_api_tool,news_archive_tool],
        llm=ClaudeSonnet
    )

    academic_curator = Agent(
        role="Academic Source Curator",
        goal="Validate and synthesize academic sources on South Asian history, identity politics, and religious minorities",
        backstory="""Former research librarian at Oxford's Bodleian Libraries, specializing in South Asian studies. Developed innovative citation analysis methods to track scholarly discourse on minority communities. Led digital humanities initiatives connecting historical archives across India, Bangladesh, and Pakistan. Known for uncovering overlooked primary sources that challenge dominant historical narratives.""",
        tools=[google_scholar_tool, search_api_tool],
        llm=ClaudeSonnet
    )

    historical_analysis = Task(
        description=f"Analyze historical development and transformation of {research_topic}, focusing on cultural shifts and community experiences",
        agent=historical_analyst,
        expected_output="Detailed historical analysis with verified academic sources"
    )
    
    media_analysis = Task(
        description="Analyze media representation patterns, focusing on language, framing, and narrative evolution",
        agent=media_analyzer,
        expected_output="Media analysis report with source documentation",
        context=[historical_analysis]
    )
    
    source_curation = Task(
        description="Compile and validate academic sources supporting the analysis",
        agent=academic_curator,
        expected_output="Annotated bibliography with citation metrics",
        context=[historical_analysis, media_analysis]
    )

    return [historical_analyst, media_analyzer, academic_curator], [historical_analysis, media_analysis, source_curation]

def run_research(research_topic, progress_containers):
    agents, tasks = create_agents_and_tasks(research_topic)
    
    # Initialize all containers with waiting state
    progress_containers["historical"].info("⏳ Waiting to start historical analysis...")
    progress_containers["media"].info("⏳ Waiting to start media analysis...")
    progress_containers["sources"].info("⏳ Waiting to start source curation...")
    
    # Start Historical Analysis
    with progress_containers["historical"].status("🔄 Analyzing historical context..."):
        progress_containers["historical"].write("This may take a few minutes...")
    
    # Store user's research topic in Mem0
    messages = [
        {"role": "user", "content": f"Research topic: {research_topic}"},
        {"role": "assistant", "content": f"I'll help you research about {research_topic} in South Asian history."}
    ]
    mem0_client.add(messages, user_id=st.session_state.get("user_id", "default_user"))
    
    crew = Crew(
        agents=agents,
        tasks=tasks,
        verbose=True,
        process=Process.sequential,
        memory=False
    )

    def process_output(output):
        # Split the output based on agent sections
        sections = output.split("# Agent:")
        
        for section in sections[1:]:  # Skip the first empty section
            if "Historical Analyst" in section:
                progress_containers["historical"].success("✅ Historical Analysis Complete")
                progress_containers["historical"].markdown(section.split("# Agent:")[0].strip())
                # Start Media Analysis
                progress_containers["media"].status("🔄 Analyzing media representations...")
                progress_containers["media"].write("This may take a few minutes...")
            elif "Media Content Analyzer" in section:
                progress_containers["media"].success("✅ Media Analysis Complete")
                progress_containers["media"].markdown(section.split("# Agent:")[0].strip())
                # Start Source Curation
                progress_containers["sources"].status("🔄 Curating academic sources...")
                progress_containers["sources"].write("This may take a few minutes...")
            elif "Academic Source Curator" in section:
                progress_containers["sources"].success("✅ Source Curation Complete")
                progress_containers["sources"].markdown(section.split("# Agent:")[0].strip())
        
        return output

    output = crew.kickoff()
    # Convert CrewOutput to string before processing
    output_str = str(output)
    return process_output(output_str)

# Initialize session state for user ID if not exists
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{os.urandom(4).hex()}"

# Streamlit UI
st.title("📚 South Asian History Research")
st.write("Analyze historical topics in South Asian history with AI-powered research agents!")

# Input section
with st.form("research_form"):
    research_topic = st.text_input("Enter your research topic:", 
                               placeholder="e.g., Cultural transformation in Assam")
    submitted = st.form_submit_button("Start Research")

# Process and display results
if submitted and research_topic:
    st.write("Starting research process...")
    
    try:
        # Create containers for each step
        progress_containers = {
            "historical": st.container(),
            "media": st.container(),
            "sources": st.container()
        }
        
        # Add headers for each section
        with progress_containers["historical"]:
            st.subheader("1️⃣ Historical Analysis")
        with progress_containers["media"]:
            st.subheader("2️⃣ Media Analysis")
        with progress_containers["sources"]:
            st.subheader("3️⃣ Academic Sources")
        
        st.write(f"Researching topic: {research_topic}")
        
        # Initialize progress indicators
        progress_containers["historical"].info("⏳ Initializing historical analysis...")
        progress_containers["media"].info("⏳ Waiting to start media analysis...")
        progress_containers["sources"].info("⏳ Waiting to start source curation...")
        
        result = run_research(research_topic, progress_containers)
        
        st.success("Research completed!")
        
        # Create download button
        st.download_button(
            label="Download Research Results",
            data=result,
            file_name=f"research_results_{research_topic.replace(' ', '_')}.txt",
            mime="text/plain"
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write(f"Error details: {str(e)}")
        st.write("Please try again with a different topic or contact support if the issue persists.")

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use:
1. Enter your South Asian history research topic in the text field
2. Click 'Start Research' to begin the analysis
3. Wait for the AI agents to complete their research tasks
4. Download the comprehensive research results

The AI will provide:
- Detailed historical analysis
- Media representation analysis
- Curated academic sources with citations
""")
