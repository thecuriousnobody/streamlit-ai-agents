import streamlit as st
import os
import time
import sys
import traceback
from datetime import datetime
from crewai import Agent, Task, Crew, Process, LLM

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

# Initialize session state
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
if 'task_outputs' not in st.session_state:
    st.session_state.task_outputs = {}
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = []
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'task_progress' not in st.session_state:
    st.session_state.task_progress = 0

def log_debug_info(message, level="info"):
    """Add timestamped debug information"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    info = {
        "timestamp": timestamp,
        "message": message,
        "level": level
    }
    st.session_state.debug_info.append(info)

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
            expected_output="Numbered list of 7-10 findings with specific dates and evidence"
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
            expected_output="Numbered list of 7-10 findings with examples and impacts",
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
        return [research_analyst, policy_media_analyst, source_curator], tasks
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
    st.session_state.research_status = "🔄 Conducting research analysis..."
    st.session_state.policy_media_status = "⏳ Waiting to start policy and media analysis..."
    st.session_state.sources_status = "⏳ Waiting to start source curation..."
    st.session_state.start_time = time.time()
    st.session_state.task_progress = 0
    
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
                max_rpm=30
            )
            
            output = crew.kickoff()
            
            if output and str(output).strip():
                output_str = str(output)
                
                # Try to process output sections if possible
                try:
                    sections = output_str.split("# Agent:")
                    if len(sections) > 1:
                        for section in sections[1:]:
                            if "Research Analyst" in section:
                                st.session_state.research_status = "✅ Research Analysis Complete"
                                st.session_state.policy_media_status = "🔄 Analyzing policy and media..."
                                st.session_state.task_progress = 33
                            elif "Policy & Media Analyst" in section:
                                st.session_state.policy_media_status = "✅ Policy & Media Analysis Complete"
                                st.session_state.sources_status = "🔄 Curating sources..."
                                st.session_state.task_progress = 66
                            elif "Source Curator" in section:
                                st.session_state.sources_status = "✅ Source Curation Complete"
                                st.session_state.task_progress = 100
                    else:
                        # If we can't split by agents, just mark everything as complete
                        st.session_state.research_status = "✅ Research Analysis Complete"
                        st.session_state.policy_media_status = "✅ Policy & Media Analysis Complete"
                        st.session_state.sources_status = "✅ Source Curation Complete"
                        st.session_state.task_progress = 100
                except Exception as e:
                    # If there's any error processing sections, still continue
                    st.session_state.research_status = "✅ Research Analysis Complete"
                    st.session_state.policy_media_status = "✅ Policy & Media Analysis Complete"
                    st.session_state.sources_status = "✅ Source Curation Complete"
                    st.session_state.task_progress = 100
                    log_debug_info(f"Error processing sections: {str(e)}", "warning")
                
                # Store task outputs
                for task in tasks:
                    if hasattr(task, 'output') and task.output:
                        st.session_state.task_outputs[task.description] = {
                            'description': task.description,
                            'summary': task.output.summary if hasattr(task.output, 'summary') else None,
                            'raw': task.output.raw if hasattr(task.output, 'raw') else None
                        }
                
                # Clean up the output string
                cleaned_output = output_str.strip()
                if not cleaned_output.endswith('\n'):
                    cleaned_output += '\n'
                
                st.session_state.research_results = cleaned_output
                break
            
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2 * retry_count)
                
        except Exception as e:
            retry_count += 1
            if retry_count == max_retries:
                error_msg = f"Research process failed: {str(e)}"
                st.error(error_msg)
                st.session_state.research_status = "❌ Research Analysis Failed"
                st.session_state.policy_media_status = "❌ Policy & Media Analysis Failed"
                st.session_state.sources_status = "❌ Source Curation Failed"
                log_debug_info(f"Research failed: {str(e)}", "error")
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
        col1, col2 = st.columns([2, 1])
        with col1:
            st.progress(st.session_state.task_progress / 100, "Overall Progress")
        with col2:
            if st.session_state.start_time:
                elapsed = time.time() - st.session_state.start_time
                st.metric("⏱️ Time Elapsed", format_time(elapsed))

    # Display status
    st.header("1️⃣ Research Analysis")
    st.write(st.session_state.research_status)
    
    st.header("2️⃣ Policy & Media Analysis")
    st.write(st.session_state.policy_media_status)
    
    st.header("3️⃣ Sources & Bibliography")
    st.write(st.session_state.sources_status)

    # Display results in tabs
    if st.session_state.research_results:
        tab1, tab2 = st.tabs(["📝 Results", "📊 Analysis"])
        
        with tab1:
            st.text_area("Research Results", st.session_state.research_results, height=300)
            
            # Download button
            if st.download_button(
                "📥 Download Results",
                data=st.session_state.research_results,
                file_name=f"research_{research_topic.lower().replace(' ', '_')}.txt",
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
            
            # Display task outputs if available
            if st.session_state.task_outputs:
                st.subheader("Task Details")
                for task_desc, task_output in st.session_state.task_outputs.items():
                    with st.expander(f"Task: {task_desc[:100]}..."):
                        if task_output.get('summary'):
                            st.write("Summary:", task_output['summary'])
                        if task_output.get('raw'):
                            st.write("Raw Output:")
                            st.code(task_output['raw'])

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

    st.divider()
    
    # How to use section
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
        """)

if __name__ == "__main__":
    main()
