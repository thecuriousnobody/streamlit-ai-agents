import streamlit as st
import os
import time
from crewai import Agent, Task, Crew, Process, LLM
import sys
from datetime import datetime
import json
from pydantic import BaseModel

# Modified environment handling
def get_environment_variables():
    """Get environment variables with fallback to Streamlit secrets"""
    variables = {
        'ANTHROPIC_API_KEY': None,
        'SERPER_API_KEY': None
    }
    
    for key in variables.keys():
        # Try getting from environment first
        value = os.getenv(key)
        
        # If not in environment, try Streamlit secrets
        if not value and hasattr(st, 'secrets'):
            value = st.secrets.get(key)
            
        variables[key] = value
    
    return variables

# Initialize environment
env_vars = get_environment_variables()

# Validate required environment variables
required_vars = ['ANTHROPIC_API_KEY', 'SERPER_API_KEY']
missing_vars = [var for var in required_vars if not env_vars.get(var)]

if missing_vars:
    st.error(f"""
        Missing required environment variables: {', '.join(missing_vars)}
        
        Please set these variables in your Render dashboard:
        1. Go to your app settings in Render
        2. Navigate to the Environment section
        3. Add the required environment variables
    """)
    st.stop()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_tools_serper import serper_search_tool, serper_scholar_tool

try:
    # Initialize LLM instances
    ClaudeSonnet = LLM(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=8192,
        temperature=0.6
    )
    
    ClaudeHaiku = LLM(
        model="claude-3-5-haiku-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=8192,
        temperature=0.6
    )
    
except Exception as e:
    st.error("""
        Please set up your API keys in Streamlit Cloud:
        1. Go to your app settings in Streamlit Cloud
        2. Navigate to the Secrets section
        3. Add the following secrets:
        ```toml
        ANTHROPIC_API_KEY = "your-anthropic-api-key"
        SERPER_API_KEY = "your-searchapi-key"
        ```
    """)
    st.stop()

# Initialize environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Enhanced session state initialization
if 'research_results' not in st.session_state:
    st.session_state.research_results = ""
if 'research_status' not in st.session_state:
    st.session_state.research_status = "⏳ Waiting to start research analysis..."
if 'policy_media_status' not in st.session_state:
    st.session_state.policy_media_status = "⏳ Waiting to start policy and media analysis..."
if 'sources_status' not in st.session_state:
    st.session_state.sources_status = "⏳ Waiting to start source curation..."
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{os.urandom(4).hex()}"
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []
if 'current_task' not in st.session_state:
    st.session_state.current_task = ""
if 'task_progress' not in st.session_state:
    st.session_state.task_progress = 0
if 'agent_thoughts' not in st.session_state:
    st.session_state.agent_thoughts = []
if 'tool_uses' not in st.session_state:
    st.session_state.tool_uses = []
if 'current_agent' not in st.session_state:
    st.session_state.current_agent = ""
if 'estimated_time' not in st.session_state:
    st.session_state.estimated_time = None

def log_debug_info(message, level="info"):
    """Add timestamped debug information"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.debug_info.append({
        "timestamp": timestamp,
        "message": message,
        "level": level
    })

def log_agent_thought(agent_name, thought):
    """Log agent's current thought process"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.agent_thoughts.append({
        "timestamp": timestamp,
        "agent": agent_name,
        "thought": thought
    })
    st.session_state.current_agent = agent_name

def log_tool_use(tool_name, input_data):
    """Log tool usage with timestamp"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.tool_uses.append({
        "timestamp": timestamp,
        "tool": tool_name,
        "input": input_data
    })

def update_progress(stage, progress, message=None):
    """Update progress based on current stage"""
    if stage == "research":
        st.session_state.task_progress = progress
        if message:
            st.session_state.research_status = message
        elif progress == 100:
            st.session_state.research_status = "✅ Research Analysis Complete"
    elif stage == "policy":
        st.session_state.task_progress = 33 + (progress * 0.33)
        if message:
            st.session_state.policy_media_status = message
        elif progress == 100:
            st.session_state.policy_media_status = "✅ Policy & Media Analysis Complete"
    elif stage == "sources":
        st.session_state.task_progress = 66 + (progress * 0.34)
        if message:
            st.session_state.sources_status = message
        elif progress == 100:
            st.session_state.sources_status = "✅ Source Curation Complete"

class ProgressCallback:
    """Callback handler for tracking agent progress"""
    def on_tool_start(self, agent_name, tool_name, input_data):
        log_tool_use(tool_name, input_data)
        if "Research Analyst" in agent_name:
            update_progress("research", 25, "🔍 Searching for historical data...")
        elif "Policy & Media Analyst" in agent_name:
            update_progress("policy", 25, "🔍 Analyzing policy documents...")
        elif "Source Curator" in agent_name:
            update_progress("sources", 25, "🔍 Finding academic sources...")
        st.experimental_rerun()
        
    def on_agent_start(self, agent_name):
        st.session_state.current_agent = agent_name
        if "Research Analyst" in agent_name:
            update_progress("research", 10, "🔄 Starting research analysis...")
        elif "Policy & Media Analyst" in agent_name:
            update_progress("policy", 10, "🔄 Starting policy analysis...")
        elif "Source Curator" in agent_name:
            update_progress("sources", 10, "🔄 Starting source curation...")
        st.experimental_rerun()
        
    def on_agent_end(self, agent_name):
        if "Research Analyst" in agent_name:
            update_progress("research", 100)
        elif "Policy & Media Analyst" in agent_name:
            update_progress("policy", 100)
        elif "Source Curator" in agent_name:
            update_progress("sources", 100)
        st.experimental_rerun()

def create_agents_and_tasks(research_topic):
    """Create research agents and tasks."""
    try:
        log_debug_info("Creating research agents...")
        research_analyst = Agent(
            role="Research Analyst",
            goal="""Conduct comprehensive historical and ethnographic analysis of South Asian topics,
                    focusing on cultural transformation, community experiences, and social dynamics""",
            backstory="""Expert historian and ethnographer specializing in South Asian studies with extensive 
                        fieldwork experience. Known for combining historical analysis with contemporary 
                        cultural understanding to provide deep insights into community dynamics and social transformation.""",
            tools=[serper_scholar_tool, serper_search_tool],
            llm=ClaudeHaiku,
            verbose=True
        )

        policy_media_analyst = Agent(
            role="Policy & Media Analyst",
            goal="""Analyze policy implementations and media representations, examining their impact
                    on communities and public discourse""",
            backstory="""Experienced policy researcher and media analyst specializing in South Asian affairs.
                        Expert in analyzing government policies, legal frameworks, and their representation
                        in various media formats. Known for connecting policy decisions to their real-world
                        impacts on communities.""",
            tools=[serper_search_tool],
            llm=ClaudeHaiku,
            verbose=True
        )

        source_curator = Agent(
            role="Source & Citation Specialist",
            goal="""Create a comprehensive research foundation by finding, validating, and properly citing key sources.""",
            backstory="""Research librarian and citation specialist with expertise in South Asian studies. 
                        Known for creating meticulous source documentation and finding precise supporting evidence for academic claims.""",
            tools=[serper_scholar_tool, serper_search_tool],
            llm=ClaudeHaiku,
            verbose=True
        )

        log_debug_info("Creating research tasks...")
        research_analysis = Task(
            description=f"""Analyze {research_topic} and provide exactly 5-7 key findings:
                - 2 major historical developments/events with dates
                - 2 significant cultural transformations with concrete examples
                - 2 key community experiences/perspectives with evidence
                Format each finding in 2-3 sentences maximum.
                Total response should not exceed 750 words.""",
            agent=research_analyst,
            expected_output="Numbered list of 5-7 findings with specific dates and evidence"
        )

        policy_media_analysis = Task(
            description=f"""Analyze {research_topic} and provide exactly 5-7 key findings:
                - 2 critical policy developments and their implementation impacts
                - 2 dominant media narratives with specific examples
                - 2 key public discourse patterns with evidence
                Format each finding in 2-3 sentences maximum.
                Include specific dates, sources, or examples for each finding.
                Total response should not exceed 750 words.""",
            agent=policy_media_analyst,
            expected_output="Numbered list of 5-7 findings with examples and impacts",
            context=[research_analysis]
        )

        source_curation = Task(
            description="""Compile exactly 6 key sources with complete citation information:
                
                For each source provide:
                1. Full academic citation (Chicago style)
                2. Direct link to the source (DOI, Google Scholar, or stable URL if available)
                3. 1-2 key quotes that support specific research claims
                4. Publication impact metrics (citation count, journal ranking)
                5. Brief note on source credibility/authority
                
                Organize sources into:
                - 2 foundational academic sources (peer-reviewed journals/books)
                - 2 primary sources (government documents, legal texts, archival materials)
                - 2 contemporary sources (recent scholarship, current analyses)
                
                Total response not to exceed 1000 words.
                Each source must include all citation elements for proper academic reference.""",
            agent=source_curator,
            expected_output="Structured bibliography with 6 fully cited sources",
            context=[research_analysis, policy_media_analysis]
        )

        tasks = [research_analysis, policy_media_analysis, source_curation]

        log_debug_info("Agents and tasks created successfully")
        return [research_analyst, policy_media_analyst, source_curator], [
            research_analysis,
            policy_media_analysis,
            source_curation
        ]
    except Exception as e:
        log_debug_info(f"Error creating agents: {str(e)}", "error")
        st.error(f"Error creating agents: {str(e)}")
        return None, None

def start_research(research_topic):
    """Start the research process."""
    if not research_topic:
        st.error("Please enter a research topic")
        return
        
    st.session_state.is_processing = True
    st.session_state.error_message = ""
    st.session_state.start_time = time.time()
    st.session_state.debug_info = []
    st.session_state.agent_thoughts = []
    st.session_state.tool_uses = []
    st.session_state.task_progress = 0
    st.session_state.estimated_time = 180  # Initial estimate: 3 minutes
    
    log_debug_info(f"Starting research on topic: {research_topic}")
    st.session_state.research_status = "🔄 Conducting research analysis..."
    st.session_state.policy_media_status = "⏳ Waiting to start policy and media analysis..."
    st.session_state.sources_status = "⏳ Waiting to start source curation..."
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create and run crew
            agents, tasks = create_agents_and_tasks(research_topic)
            if not agents or not tasks:
                st.session_state.is_processing = False
                return
                
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=True,
                process=Process.sequential,
                memory=False,
                max_rpm=30,
                callback_handler=ProgressCallback()
            )
            
            log_debug_info("Starting crew execution")
            output = crew.kickoff()
            
            if output and str(output).strip():
                output_str = str(output)
                
                # Clean up the output string
                cleaned_output = output_str.strip()
                if not cleaned_output.endswith('\n'):
                    cleaned_output += '\n'
                
                st.session_state.research_results = cleaned_output
                log_debug_info("Research process completed successfully")
                break
            
            retry_count += 1
            if retry_count < max_retries:
                log_debug_info(f"Retrying... Attempt {retry_count + 1} of {max_retries}")
                time.sleep(2 * retry_count)
                
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                error_msg = f"Research process failed: {str(e)}"
                st.error(error_msg)
                log_debug_info(error_msg, "error")
                st.session_state.research_status = "❌ Research Analysis Failed"
                st.session_state.policy_media_status = "❌ Policy & Media Analysis Failed"
                st.session_state.sources_status = "❌ Source Curation Failed"
            time.sleep(2 * retry_count)
            
    st.session_state.is_processing = False

def format_time(seconds):
    """Format time in seconds to a readable string"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes}m {seconds}s"

def main():
    st.title("📚 South Asian History Research Pro")
    st.write("Enhanced research analysis with detailed progress tracking and debug information!")

    # Research topic input
    research_topic = st.text_input("Enter your research topic", placeholder="e.g., Cultural transformation in Assam")
    
    # Start research button
    if st.button("Start Research", disabled=st.session_state.is_processing):
        start_research(research_topic)

    # Display error message if any
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    st.divider()

    # Display overall progress with metrics
    if st.session_state.is_processing or st.session_state.task_progress > 0:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.progress(st.session_state.task_progress / 100, "Overall Progress")
        with col2:
            if st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                st.metric("⏱️ Elapsed Time", format_time(elapsed))
        with col3:
            if st.session_state.estimated_time:
                remaining = max(0, st.session_state.estimated_time - (time.time() - st.session_state.start_time))
                st.metric("⏳ Estimated Time", format_time(remaining))

    # Current Agent Activity
    if st.session_state.current_agent:
        st.subheader("🤖 Current Activity")
        st.info(f"Agent: {st.session_state.current_agent}")
        
        # Show latest thought if available
        if st.session_state.agent_thoughts:
            latest_thought = st.session_state.agent_thoughts[-1]
            st.write(f"💭 Current thought: {latest_thought['thought']}")
        
        # Show latest tool use if available
        if st.session_state.tool_uses:
            latest_tool = st.session_state.tool_uses[-1]
            with st.expander("🔧 Latest Tool Use"):
                st.write(f"Tool: {latest_tool['tool']}")
                st.write(f"Input: {latest_tool['input']}")

    # Display detailed status with spinners
    col1, col2 = st.columns(2)
    with col1:
        st.header("Research Progress")
        
        # Research Analysis
        status_col1, spinner_col1 = st.columns([3, 1])
        with status_col1:
            st.write("1️⃣ Research Analysis")
            st.write(st.session_state.research_status)
        with spinner_col1:
            if "🔄" in st.session_state.research_status:
                with st.spinner("Analyzing..."):
                    pass
        
        # Policy & Media Analysis
        status_col2, spinner_col2 = st.columns([3, 1])
        with status_col2:
            st.write("2️⃣ Policy & Media Analysis")
            st.write(st.session_state.policy_media_status)
        with spinner_col2:
            if "🔄" in st.session_state.policy_media_status:
                with st.spinner("Analyzing..."):
                    pass
        
        # Sources & Bibliography
        status_col3, spinner_col3 = st.columns([3, 1])
        with status_col3:
            st.write("3️⃣ Sources & Bibliography")
            st.write(st.session_state.sources_status)
        with spinner_col3:
            if "🔄" in st.session_state.sources_status:
                with st.spinner("Curating..."):
                    pass
    
    with col2:
        st.header("Activity Log")
        # Show recent agent thoughts
        if st.session_state.agent_thoughts:
            with st.expander("💭 Recent Thoughts", expanded=True):
                for thought in reversed(st.session_state.agent_thoughts[-5:]):
                    st.write(f"**{thought['timestamp']}** - {thought['agent']}")
                    st.write(f"_{thought['thought']}_")
        
        # Show recent tool uses
        if st.session_state.tool_uses:
            with st.expander("🔧 Recent Tool Uses", expanded=True):
                for tool in reversed(st.session_state.tool_uses[-5:]):
                    st.write(f"**{tool['timestamp']}** - {tool['tool']}")
                    st.write(f"Input: _{tool['input']}_")

    # Debug Information (Collapsible)
    if st.session_state.debug_info:
        with st.expander("🔍 Debug Information"):
            for info in st.session_state.debug_info:
                if info["level"] == "error":
                    st.error(f"{info['timestamp']}: {info['message']}")
                elif info["level"] == "warning":
                    st.warning(f"{info['timestamp']}: {info['message']}")
                else:
                    st.info(f"{info['timestamp']}: {info['message']}")

    def get_filename_from_topic(topic):
        if not topic:
            return "research_results.txt"
        
        # Common words to exclude
        stop_words = {'a', 'an', 'the', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'by', 'with', 'about'}
        
        # Split into words and convert to lowercase
        words = topic.lower().split()
        
        # Keep only significant words
        key_words = [word for word in words if word not in stop_words]
        
        # Take first 2-3 significant words
        selected_words = key_words[:3]
        
        # If we have less than 2 words, use what we have
        if len(selected_words) == 0:
            return "research_results.txt"
            
        # Join words with underscores
        filename = f"research_{('_'.join(selected_words))}.txt"
        
        return filename

    # Display results in tabs
    if st.session_state.research_results:
        tab1, tab2 = st.tabs(["📝 Results", "📊 Analysis"])
        
        with tab1:
            st.text_area("Research Results", st.session_state.research_results, height=300)
            
            # Download button
            if st.download_button(
                "📥 Download Results",
                data=st.session_state.research_results,
                file_name=get_filename_from_topic(research_topic),
                mime="text/plain"
            ):
                st.success("File downloaded successfully!")
        
        with tab2:
            # Display some basic analytics about the results
            results = st.session_state.research_results
            st.write("📊 Results Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", len(results.split()))
            with col2:
                st.metric("Character Count", len(results))
            with col3:
                st.metric("Section Count", len(results.split("# Agent:")))

    st.divider()
    
    # How to use section in an expander
    with st.expander("ℹ️ How to use"):
        st.markdown("""
            ### Getting Started:
            1. Enter your South Asian history research topic in the text field
            2. Click 'Start Research' to begin the analysis
            3. Monitor real-time progress with detailed status updates
            4. Download comprehensive research results when complete

            ### What You'll Get:
            - 📚 Historical and cultural analysis
            - 📰 Policy impact and media representation analysis
            - 📑 Curated academic sources with citations
            
            ### Pro Features:
            - ⏱️ Real-time progress tracking
            - 📊 Results analytics
            - 🔍 Debug information
            - 💾 Formatted downloads
            - 🔄 Live activity monitoring
        """)

if __name__ == "__main__":
    main()
