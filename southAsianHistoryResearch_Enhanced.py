import streamlit as st
import os
import time
import sys
import traceback
from datetime import datetime
import json
from pydantic import BaseModel
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

# Custom error handling
class ResearchError(Exception):
    """Custom error class for research operations"""
    def __init__(self, message: str, phase: str, details: dict = None):
        self.message = message
        self.phase = phase
        self.details = details
        super().__init__(self.message)

def handle_research_error(error: ResearchError, status_container):
    """Handle research errors and update UI accordingly"""
    # Update status container
    status_container.update(
        label=f"Error in {error.phase}",
        state="error",
        expanded=True
    )
    
    # Show error details
    with status_container:
        st.error(error.message)
        if error.details:
            st.write("Error Details:")
            st.json(error.details)
    
    # Cache error state
    cache_research_state(
        st.session_state.user_id,
        {
            'error': {
                'phase': error.phase,
                'message': error.message,
                'details': error.details
            }
        }
    )

# Cache management
@st.cache_data
def cache_research_state(session_id: str, state_data: dict):
    """Cache the current state of research"""
    return {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'state': state_data
    }

@st.cache_data
def cache_agent_activity(session_id: str, agent_name: str, activity: dict):
    """Cache individual agent activities"""
    return {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'agent': agent_name,
        'activity': activity
    }

@st.cache_data
def get_cached_research_state(session_id: str):
    """Retrieve cached research state"""
    return None  # Will be populated during research

# Enhanced session state initialization
def initialize_session_state():
    """Initialize or reset session state variables"""
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"user_{os.urandom(4).hex()}"
        
    # Research state
    st.session_state.research_results = ""
    st.session_state.is_processing = False
    st.session_state.error_message = ""
    st.session_state.start_time = None
    st.session_state.estimated_time = 180  # Initial estimate: 3 minutes
    
    # Progress tracking
    st.session_state.task_progress = 0
    st.session_state.current_task = ""
    st.session_state.current_agent = ""
    
    # Status messages
    st.session_state.research_status = "⏳ Waiting to start research analysis..."
    st.session_state.policy_media_status = "⏳ Waiting to start policy and media analysis..."
    st.session_state.sources_status = "⏳ Waiting to start source curation..."
    
    # Activity logging
    st.session_state.debug_info = []
    st.session_state.agent_thoughts = []
    st.session_state.tool_uses = []
    st.session_state.tool_results = []
    
    # Task tracking
    st.session_state.current_tasks = {}
    st.session_state.task_outputs = {}

# Initialize session state
initialize_session_state()

# Enhanced logging functions
def log_debug_info(message, level="info"):
    """Add timestamped debug information with caching"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    info = {
        "timestamp": timestamp,
        "message": message,
        "level": level
    }
    st.session_state.debug_info.append(info)
    
    # Cache debug info
    cache_research_state(
        st.session_state.user_id,
        {'debug_info': st.session_state.debug_info}
    )

def log_agent_thought(agent_name, thought):
    """Log agent's thought process with caching"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    thought_data = {
        "timestamp": timestamp,
        "agent": agent_name,
        "thought": thought
    }
    st.session_state.agent_thoughts.append(thought_data)
    st.session_state.current_agent = agent_name
    
    # Cache agent thought
    cache_agent_activity(
        st.session_state.user_id,
        agent_name,
        {'thought': thought_data}
    )

def log_tool_use(tool_name, input_data):
    """Log tool usage with caching"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    tool_data = {
        "timestamp": timestamp,
        "tool": tool_name,
        "input": input_data
    }
    st.session_state.tool_uses.append(tool_data)
    
    # Cache tool usage
    cache_agent_activity(
        st.session_state.user_id,
        st.session_state.current_agent,
        {'tool_use': tool_data}
    )

# Enhanced progress tracking
def update_progress(stage, progress, message=None):
    """Update and cache progress for each stage"""
    # Calculate progress and update status
    if stage == "research":
        st.session_state.task_progress = progress
        status_message = message if message else "✅ Research Analysis Complete" if progress == 100 else st.session_state.research_status
        st.session_state.research_status = status_message
    elif stage == "policy":
        st.session_state.task_progress = 33 + (progress * 0.33)
        status_message = message if message else "✅ Policy & Media Analysis Complete" if progress == 100 else st.session_state.policy_media_status
        st.session_state.policy_media_status = status_message
    elif stage == "sources":
        st.session_state.task_progress = 66 + (progress * 0.34)
        status_message = message if message else "✅ Source Curation Complete" if progress == 100 else st.session_state.sources_status
        st.session_state.sources_status = status_message
    
    # Cache progress state
    cache_research_state(
        st.session_state.user_id,
        {
            'stage': stage,
            'progress': progress,
            'status': status_message,
            'overall_progress': st.session_state.task_progress
        }
    )

class ProgressCallback:
    """Enhanced callback handler for tracking agent progress with real-time updates"""
    
    def __init__(self):
        self.session_id = st.session_state.user_id
        self.start_time = time.time()
        if 'tool_results' not in st.session_state:
            st.session_state.tool_results = []
        
    def on_tool_start(self, agent_name, tool_name, input_data):
        """Handle tool start with enhanced tracking and result display"""
        # Log and cache activity
        log_tool_use(tool_name, input_data)
        
        # Display tool execution status
        st.write(f"🔄 {agent_name} using {tool_name}")
        st.write(f"Input: {input_data}")
        
        # Cache activity
        cache_agent_activity(
            self.session_id,
            agent_name,
            {
                'type': 'tool_start',
                'tool': tool_name,
                'input': input_data,
                'timestamp': time.time()
            }
        )
        
        # Update progress based on agent
        if "Research Analyst" in agent_name:
            update_progress("research", 25, "🔍 Searching for historical data...")
        elif "Policy & Media Analyst" in agent_name:
            update_progress("policy", 25, "🔍 Analyzing policy documents...")
        elif "Source Curator" in agent_name:
            update_progress("sources", 25, "🔍 Finding academic sources...")
    
    def on_tool_end(self, agent_name, tool_name, output):
        """Handle tool completion with result display"""
        # Store tool result
        result = {
            'agent': agent_name,
            'tool': tool_name,
            'output': output,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        }
        st.session_state.tool_results.append(result)
        
        # Display result
        st.write(f"✅ {agent_name} completed {tool_name}")
        st.write("Tool Result:")
        st.write(output)
    
    def on_agent_start(self, agent_name):
        """Handle agent start with enhanced tracking and status updates"""
        st.session_state.current_agent = agent_name
        
        # Display agent start
        st.write(f"🤖 {agent_name} Starting Work")
        st.write(f"Agent initialized at {datetime.now().strftime('%H:%M:%S')}")
        
        # Cache agent start
        cache_agent_activity(
            self.session_id,
            agent_name,
            {
                'type': 'agent_start',
                'timestamp': time.time()
            }
        )
        
        # Update progress
        if "Research Analyst" in agent_name:
            update_progress("research", 10, "🔄 Starting research analysis...")
        elif "Policy & Media Analyst" in agent_name:
            update_progress("policy", 10, "🔄 Starting policy analysis...")
        elif "Source Curator" in agent_name:
            update_progress("sources", 10, "🔄 Starting source curation...")
    
    def on_agent_end(self, agent_name, task=None):
        """Handle agent completion with enhanced result display"""
        # Calculate duration
        duration = time.time() - self.start_time
        
        # Display completion status
        st.write(f"✅ {agent_name} Complete")
        st.write(f"Completed in {format_time(duration)}")
            
        # Show task output if available
        if task and hasattr(task, 'output'):
            # Store task output in session state
            st.session_state.task_outputs[task.description] = task.output
            
            # Display task output
            st.markdown("### Task Details")
            st.write(f"**Description:** {task.output.description}")
            if task.output.summary:
                st.write(f"**Summary:** {task.output.summary}")
            
            st.markdown("### Results")
            # Display raw output with formatting
            st.markdown("""
            ```markdown
            {}
            ```
            """.format(task.output.raw))
            
            # Show any JSON output if available
            if hasattr(task.output, 'json_dict') and task.output.json_dict:
                st.write("JSON Output:")
                st.json(task.output.json_dict)
                
                # Update research results immediately
                if 'research_results' not in st.session_state:
                    st.session_state.research_results = ""
                st.session_state.research_results += f"\n\n# {agent_name}\n{task.output.raw}"
                
                # Display current results in the task output container
                task_output_container = st.empty()
                task_output_container.text_area(
                    "Current Results",
                    st.session_state.research_results,
                    height=300
                )
        
        # Show tool usage summary
        if st.session_state.tool_results:
            st.write("🔧 Tool Usage Summary:")
            for result in st.session_state.tool_results:
                if result['agent'] == agent_name:
                    st.write(f"**{result['tool']}** at {result['timestamp']}")
                    st.write(result['output'])
        
        # Cache completion with task output
        cache_data = {
            'type': 'agent_end',
            'duration': duration,
            'timestamp': time.time()
        }
        if task and hasattr(task, 'output'):
            cache_data['task_output'] = {
                'description': task.output.description,
                'summary': task.output.summary if task.output.summary else None,
                'raw': task.output.raw
            }
        cache_agent_activity(self.session_id, agent_name, cache_data)
        
        # Update progress
        if "Research Analyst" in agent_name:
            update_progress("research", 100)
        elif "Policy & Media Analyst" in agent_name:
            update_progress("policy", 100)
        elif "Source Curator" in agent_name:
            update_progress("sources", 100)

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
        
        # Create task output infrastructure
        task_output = {
            'description': research_analysis.description,
            'summary': None,
            'raw': None,
            'json_dict': None
        }
        research_analysis.output = task_output
        st.session_state.task_outputs[research_analysis.description] = task_output

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
    """Start the research process with enhanced progress tracking."""
    if not research_topic:
        st.error("Please enter a research topic")
        return
        
    # Initialize research state
    initialize_session_state()
    st.session_state.is_processing = True
    st.session_state.start_time = time.time()
    
    # Create main status container
    with st.status("🔄 Research in Progress", expanded=True) as status:
        try:
            # Create callback handler
            callback_handler = ProgressCallback()
            
            # Create agents and tasks
            agents, tasks = create_agents_and_tasks(research_topic)
            if not agents or not tasks:
                raise ResearchError("Failed to create agents and tasks", "initialization")
            
            # Store task outputs in session state
            for task in tasks:
                st.session_state.task_outputs[task.description] = task.output
            
            # Execute research with callback
            st.write("🔄 Starting research analysis...")
            
            # Create container for research output
            research_container = st.empty()
            
            try:
                # Create and execute crew with all tasks
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    process=Process.sequential,
                    verbose=True,
                    callbacks=callback_handler
                )
                
                # Execute all tasks
                results = crew.kickoff()
                
                # Process and display results
                with research_container:
                    st.markdown("### Research Results")
                    combined_results = ""
                    
                    # Process results from each task
                    for task in tasks:
                        if hasattr(task, 'output') and task.output and task.output.raw:
                            st.markdown(f"#### {task.agent.role} Findings")
                            st.markdown("""
                            ```markdown
                            {}
                            ```
                            """.format(task.output.raw))
                            
                            # Append to combined results
                            combined_results += f"\n\n# {task.agent.role}\n{task.output.raw}"
                    
                    if combined_results:
                        # Store in session state
                        st.session_state.research_results = combined_results
                        
                        # Cache final results
                        cache_research_state(
                            st.session_state.user_id,
                            {
                                'status': 'complete',
                                'result': combined_results
                            }
                        )
                        
                        # Update status to complete
                        status.update(
                            label="✅ Research Complete",
                            state="complete",
                            expanded=False
                        )
                    else:
                        raise ResearchError("No research output generated", "analysis")
            except Exception as e:
                st.error(f"Error during research analysis: {str(e)}")
                raise
                    
        except Exception as e:
            error = ResearchError(
                message=str(e),
                phase="research",
                details={'traceback': traceback.format_exc()}
            )
            handle_research_error(error, status)
            raise
        finally:
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

    # Display detailed status with spinners and results
    col1, col2 = st.columns(2)
    with col1:
        st.header("Research Progress")
        
        # Tool Results Container
        if 'tool_results' in st.session_state and st.session_state.tool_results:
            with st.expander("🔍 Latest Search Results", expanded=True):
                for result in reversed(st.session_state.tool_results[-3:]):  # Show last 3 results
                    st.markdown(f"""
                    **{result['agent']} using {result['tool']}** at {result['timestamp']}
                    ```
                    {result['output'][:500]}{'...' if len(result['output']) > 500 else ''}
                    ```
                    """)
        
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
