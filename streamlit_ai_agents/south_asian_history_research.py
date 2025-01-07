import streamlit as st
import os
import logging
import time
from crewai import Agent, Task, Crew, Process, LLM
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_tools import (
    search_api_tool,
    google_scholar_tool,
    news_archive_tool,
    local_archives_tool,
    legal_database_tool,
    government_archives_tool
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]
    
    # Initialize LLM instances
    ClaudeSonnet = LLM(
        model="claude-3-5-sonnet-20241022",
        api_key=anthropic_api_key,
        max_tokens=8192,
        temperature=0.6
    )
    
    ClaudeHaiku = LLM(
        model="claude-3-5-haiku-20241022",
        api_key=anthropic_api_key,
        max_tokens=8192,
        temperature=0.6
    )
    ClaudeOpus = LLM(
        model="claude-3-opus-20240229",
        api_key=anthropic_api_key,
        max_tokens=4096,
        temperature=0.6
    )
    LLama70b = LLM(
        api_key=groq_api_key,
        model="llama-3.3-70b-versatile",
        provider="groq",
        temperature=0.7
    )

except FileNotFoundError:
    st.error("""
        Please set up your API keys in Streamlit Cloud:
        1. Go to your app settings in Streamlit Cloud
        2. Navigate to the Secrets section
        3. Add the following secrets:
        ```toml
        ANTHROPIC_API_KEY = "your-anthropic-api-key"
        SEARCH_API_KEY = "your-searchapi-key"
        MEM0_API_KEY = "your-mem0-api-key"
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
if 'historical_status' not in st.session_state:
    st.session_state.historical_status = "⏳ Waiting to start historical analysis..."
if 'ethnographic_status' not in st.session_state:
    st.session_state.ethnographic_status = "⏳ Waiting to start ethnographic analysis..."
if 'policy_status' not in st.session_state:
    st.session_state.policy_status = "⏳ Waiting to start policy impact analysis..."
if 'media_status' not in st.session_state:
    st.session_state.media_status = "⏳ Waiting to start media analysis..."
if 'sources_status' not in st.session_state:
    st.session_state.sources_status = "⏳ Waiting to start source curation..."
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""
if 'user_id' not in st.session_state:
    st.session_state.user_id = f"user_{os.urandom(4).hex()}"

def create_agents_and_tasks(research_topic, fallback_llm=None):
    """Create research agents and tasks.
    
    Args:
        research_topic: The topic to research
        fallback_llm: If provided, this LLM will be used instead of the default Claude models
    """
    try:
        historical_analyst = Agent(
            role="Historical Analyst",
            goal="Analyze historical development of Assamese Hindu culture and Muslim communities",
            backstory="Expert in South Asian history specializing in cultural transformation and minority experiences",
            tools=[google_scholar_tool, search_api_tool],
            llm=ClaudeSonnet if fallback_llm is None else fallback_llm
        )

        media_analyzer = Agent(
            role="Media Content Analyzer",
            goal="Track and analyze media representations of Muslim communities in Assam",
            backstory="Media studies expert focusing on minority representation and narrative analysis",
            tools=[search_api_tool, news_archive_tool],
            llm=ClaudeSonnet if fallback_llm is None else fallback_llm
        )

        academic_curator = Agent(
            role="Academic Source Curator",
            goal="Validate and synthesize academic sources on South Asian history, identity politics, and religious minorities",
            backstory="""Former research librarian at Oxford's Bodleian Libraries, specializing in South Asian studies. 
            Developed innovative citation analysis methods to track scholarly discourse on minority communities. 
            Led digital humanities initiatives connecting historical archives across India, Bangladesh, and Pakistan. 
            Known for uncovering overlooked primary sources that challenge dominant historical narratives.""",
            tools=[google_scholar_tool, search_api_tool],
            llm=ClaudeHaiku if fallback_llm is None else fallback_llm
        )

        ethnographic_analyzer = Agent(
            role="Ethnographic Research Specialist",
            goal="""Analyze lived experiences and cultural narratives of both Hindu and Muslim communities in Assam through targeted multilingual searches. 
            Construct culturally-informed queries using native language terms, historical markers, and community-specific terminology to uncover authentic narratives and oral histories.""",
            backstory="""Former researcher at Centre for Northeast Studies with extensive fieldwork experience in Assam. 
            Specializes in documenting oral histories and community narratives through expert search techniques combining cultural terms in Assamese (অসমীয়া), Bengali (বাংলা), and English. 
            Published works on cultural identity and citizenship in Northeast India, known for uncovering hard-to-find community narratives by crafting precise search queries that combine temporal markers (1950-2000), 
            local terminology, and Boolean operators to access diverse archival sources. 
            Pioneered methodology for digital ethnography that leverages both contemporary and historical documentation to build comprehensive community profiles.""",
            tools=[google_scholar_tool, local_archives_tool],
            llm=ClaudeOpus if fallback_llm is None else fallback_llm
        )

        policy_analyst = Agent(
            role="Policy and Governance Analyst",
            goal="""Examine citizenship policies, NRC process, and their impacts on communities by leveraging legal databases 
            and government archives through precise legal terminology and targeted document searches across multiple jurisdictions and time periods.""",
            backstory="""Expert in Indian constitutional law and citizenship studies with deep experience in AsianLII and government documentation systems. 
            Previously worked with legal aid organizations in Assam documenting NRC cases, developing expertise in crafting searches that combine specific act numbers, 
            case references, and implementation keywords to uncover relevant legal precedents and policy documents. 
            Known for ability to trace policy evolution through comprehensive analysis of court decisions, government notifications, and academic legal commentary.""",
            tools=[legal_database_tool, government_archives_tool],
            llm=ClaudeOpus if fallback_llm is None else fallback_llm
        )

        # Tasks with enhanced sequencing
        historical_analysis = Task(
            description=f"Analyze historical development and transformation of {research_topic}, focusing on cultural shifts and community experiences",
            agent=historical_analyst,
            expected_output="Detailed historical analysis with verified academic sources"
        )

        ethnographic_analysis = Task(
            description="""Conduct comprehensive analysis of community narratives, oral histories, and lived experiences 
                        of both Hindu and Muslim communities in Assam, focusing on cultural transformation and citizenship issues""",
            agent=ethnographic_analyzer,
            expected_output="""Detailed ethnographic report including:
                            - Documented oral histories
                            - Analysis of community perspectives on cultural identity
                            - Impact assessment of Hindutva influence on local traditions
                            - Documentation of Muslim community experiences of exclusion""",
            context=[historical_analysis]
        )

        policy_impact_analysis = Task(
            description="""Analyze the implementation and impact of citizenship policies, 
                        particularly the NRC process, on different communities in Assam""",
            agent=policy_analyst,
            expected_output="""Comprehensive policy analysis including:
                            - Timeline of citizenship policy changes
                            - Case studies of NRC implementation
                            - Legal framework analysis
                            - Documentation of community-specific impacts""",
            context=[historical_analysis, ethnographic_analysis]
        )

        media_analysis = Task(
            description="Analyze media representation patterns, focusing on language, framing, and narrative evolution",
            agent=media_analyzer,
            expected_output="Media analysis report with source documentation",
            context=[historical_analysis, ethnographic_analysis, policy_impact_analysis]
        )

        source_curation = Task(
            description="""Compile and validate academic sources supporting the analysis, 
                        including ethnographic studies and policy documents""",
            agent=academic_curator,
            expected_output="""Enhanced annotated bibliography including:
                            - Academic sources
                            - Government documents
                            - Local media archives
                            - Oral history records
                            - Legal case documentation""",
            context=[historical_analysis, ethnographic_analysis, policy_impact_analysis, media_analysis]
        )

        return [
            historical_analyst, 
            ethnographic_analyzer,
            policy_analyst, 
            media_analyzer, 
            academic_curator
        ], [
            historical_analysis,
            ethnographic_analysis,
            policy_impact_analysis,
            media_analysis,
            source_curation
        ]
    except Exception as e:
        st.error(f"Error creating agents: {str(e)}")
        return None, None

def start_research(research_topic):
    """Start the research process."""
    if not research_topic:
        st.error("Please enter a research topic")
        return
        
    st.session_state.is_processing = True
    st.session_state.error_message = ""
    st.session_state.historical_status = "🔄 Analyzing historical context..."
    st.session_state.ethnographic_status = "⏳ Waiting to start ethnographic analysis..."
    st.session_state.policy_status = "⏳ Waiting to start policy impact analysis..."
    st.session_state.media_status = "⏳ Waiting to start media analysis..."
    st.session_state.sources_status = "⏳ Waiting to start source curation..."
    
    max_retries = 3
    retry_count = 0
    current_llm = None
    agents = None
    tasks = None
    
    # Define LLM fallback sequence
    llm_sequence = [
        (None, "Default Claude models"),  # None means use default LLMs
        (LLama70b, "Llama70b fallback")
    ]
    
    for fallback_llm, llm_name in llm_sequence:
        logger.info(f"Attempting research with {llm_name}")
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Optional: Store user's research topic in Mem0 if available
                mem0_api_key = os.getenv("MEM0_API_KEY")
                if mem0_api_key:
                    try:
                        from mem0 import MemoryClient
                        mem0_client = MemoryClient(api_key=mem0_api_key)
                        messages = [
                            {"role": "user", "content": f"Research topic: {research_topic}"},
                            {"role": "assistant", "content": f"I'll help you research about {research_topic} in South Asian history."}
                        ]
                        mem0_client.add(messages, user_id=st.session_state.user_id)
                    except Exception as e:
                        st.write(f"Mem0 error (non-critical): {str(e)}")
                
                # Create and run crew
                logger.info(f"Creating agents and tasks (Attempt {retry_count + 1}/{max_retries})")
                # Create agents with either default Claude models or fallback LLM
                agents, tasks = create_agents_and_tasks(research_topic, fallback_llm=fallback_llm)
                if not agents or not tasks:
                    st.session_state.is_processing = False
                    return
                    
                crew = Crew(
                    agents=agents,
                    tasks=tasks,
                    verbose=True,
                    process=Process.sequential,
                    memory=False,
                    max_rpm=30  # Limit to 30 requests per minute
                )
                
                logger.info(f"Starting crew kickoff (Attempt {retry_count + 1}/{max_retries})")
                output = crew.kickoff()
                
                if output and str(output).strip():
                    logger.info(f"Successfully received output from crew using {llm_name}")
                    output_str = str(output)
                    logger.debug("Successfully converted output to string")
                    
                    # Process output
                    sections = output_str.split("# Agent:")
                    if len(sections) <= 1:
                        raise ValueError("Invalid output format from research agents")
                        
                    for section in sections[1:]:
                        if "Historical Analyst" in section:
                            st.session_state.historical_status = "✅ Historical Analysis Complete"
                            st.session_state.ethnographic_status = "🔄 Analyzing ethnographic data..."
                        elif "Ethnographic Research Specialist" in section:
                            st.session_state.ethnographic_status = "✅ Ethnographic Analysis Complete"
                            st.session_state.policy_status = "🔄 Analyzing policy impact..."
                        elif "Policy and Governance Analyst" in section:
                            st.session_state.policy_status = "✅ Policy Impact Analysis Complete"
                            st.session_state.media_status = "🔄 Analyzing media representations..."
                        elif "Media Content Analyzer" in section:
                            st.session_state.media_status = "✅ Media Analysis Complete"
                            st.session_state.sources_status = "🔄 Curating academic sources..."
                        elif "Academic Source Curator" in section:
                            st.session_state.sources_status = "✅ Source Curation Complete"
                    
                    st.session_state.research_results = output_str
                    return  # Successfully completed
                
                # If we get an empty response, retry
                logger.warning(f"Empty response received with {llm_name}, attempt {retry_count + 1} of {max_retries}")
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 * retry_count)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                if retry_count == max_retries and fallback_llm == llm_sequence[-1][0]:
                    error_msg = f"Research process failed after exhausting all LLM options: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
                    st.session_state.historical_status = "❌ Historical Analysis Failed"
                    st.session_state.ethnographic_status = "❌ Ethnographic Analysis Failed"
                    st.session_state.policy_status = "❌ Policy Impact Analysis Failed"
                    st.session_state.media_status = "❌ Media Analysis Failed"
                    st.session_state.sources_status = "❌ Source Curation Failed"
                    break
                time.sleep(2 * retry_count)  # Exponential backoff
            
    # Make sure to reset processing state
    st.session_state.is_processing = False

def main():
    st.title("📚 South Asian History Research")
    st.write("Analyze historical topics in South Asian history with AI-powered research agents!")

    # Research topic input
    research_topic = st.text_input("Enter your research topic", placeholder="e.g., Cultural transformation in Assam")
    
    # Start research button
    if st.button("Start Research", disabled=st.session_state.is_processing):
        start_research(research_topic)

    # Display error message if any
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    st.divider()

    # Display status
    st.header("1️⃣ Historical Analysis")
    st.write(st.session_state.historical_status)
    
    st.header("2️⃣ Ethnographic Analysis")
    st.write(st.session_state.ethnographic_status)
    
    st.header("3️⃣ Policy Impact Analysis")
    st.write(st.session_state.policy_status)
    
    st.header("4️⃣ Media Analysis")
    st.write(st.session_state.media_status)
    
    st.header("5️⃣ Academic Sources")
    st.write(st.session_state.sources_status)

    # Display results
    if st.session_state.research_results:
        st.text_area("Research Results", st.session_state.research_results, height=300)
        
        # Download button
        if st.download_button(
            "Download Results",
            data=st.session_state.research_results,
            file_name=f"research_results_{research_topic.replace(' ', '_')}.txt" if research_topic else "research_results.txt",
            mime="text/plain"
        ):
            st.success("File downloaded successfully!")

    st.divider()
    
    # How to use section
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

if __name__ == "__main__":
    main()
