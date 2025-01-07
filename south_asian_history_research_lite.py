import streamlit as st
import os
import logging
import time
from crewai import Agent, Task, Crew, Process, LLM
from langchain_groq import ChatGroq
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_tools import (
    search_api_tool,
    google_scholar_tool,
    news_archive_tool,
    local_archives_tool,
    legal_database_tool
)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

api_key = os.getenv("OPENAI_API_KEY")
groq_llm = ChatGroq(
    model="groq/llama-3.1-70b-versatile",
    temperature=0.7,
    groq_api_key=st.secrets["GROQ_API_KEY"],
    verbose=True
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    anthropic_api_key = st.secrets["ANTHROPIC_API_KEY"]
    
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
    

except FileNotFoundError:
    st.error("""
        Please set up your API keys in Streamlit Cloud:
        1. Go to your app settings in Streamlit Cloud
        2. Navigate to the Secrets section
        3. Add the following secrets:
        ```toml
        ANTHROPIC_API_KEY = "your-anthropic-api-key"
        SEARCH_API_KEY = "your-searchapi-key"
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

def create_agents_and_tasks(research_topic):
    """Create research agents and tasks."""
    try:
        research_analyst = Agent(
            role="Research Analyst",
            goal="""Conduct comprehensive historical and ethnographic analysis of South Asian topics,
                    focusing on cultural transformation, community experiences, and social dynamics""",
            backstory="""Expert historian and ethnographer specializing in South Asian studies with extensive 
                        fieldwork experience. Known for combining historical analysis with contemporary 
                        cultural understanding to provide deep insights into community dynamics and social transformation.""",
            tools=[google_scholar_tool, search_api_tool, local_archives_tool],
            llm=groq_llm
        )

        policy_media_analyst = Agent(
            role="Policy & Media Analyst",
            goal="""Analyze policy implementations and media representations, examining their impact
                    on communities and public discourse""",
            backstory="""Experienced policy researcher and media analyst specializing in South Asian affairs.
                        Expert in analyzing government policies, legal frameworks, and their representation
                        in various media formats. Known for connecting policy decisions to their real-world
                        impacts on communities.""",
            tools=[legal_database_tool, news_archive_tool, search_api_tool],
            llm=groq_llm
        )

        source_curator = Agent(
            role="Source & Citation Specialist",
            goal="""Create a comprehensive research foundation by finding, validating, and properly citing key sources. 
            For each finding or claim, provide complete citation information including DOI where available, publication 
            details, and direct quotes or specific page numbers. Organize sources by theme while maintaining strict 
            academic citation standards.""",
            backstory="""Research librarian and citation specialist with expertise in South Asian studies. 
            Known for creating meticulous source documentation and finding precise supporting evidence for academic claims. 
            Expert in multiple citation formats (Chicago, APA, MLA) and skilled at tracking down primary sources and 
            their proper citations. Experienced in validating source authenticity and academic credibility.""",
            tools=[google_scholar_tool, search_api_tool],
            llm=groq_llm
        )

        research_analysis = Task(
            description=f"""Analyze {research_topic} and provide exactly 5-7 key findings:
                - 2 major historical developments/events with dates
                - 2 significant cultural transformations with concrete examples
                - 2 key community experiences/perspectives with evidence
                Format each finding in 2-3 sentences maximum.
                Total response should not exceed 750 words.""",
            agent=research_analyst,
            expected_output="""Numbered list of 5-7 findings, grouped into:
                HISTORICAL DEVELOPMENTS (1-2)
                CULTURAL TRANSFORMATIONS (3-4)
                COMMUNITY EXPERIENCES (5-6)
                Each finding includes specific dates, examples, or evidence."""
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
            expected_output="""Numbered list of 5-7 findings, grouped into:
                POLICY DEVELOPMENTS (1-2)
                MEDIA NARRATIVES (3-4)
                PUBLIC DISCOURSE (5-6)
                Each finding includes specific examples and impacts."""
        )

        source_curation = Task(
            description="""Compile exactly 6 key sources with complete citation information:
                
                For each source provide:
                1. Full academic citation (Chicago style)
                2. DOI or stable URL where available
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
            expected_output="""Structured bibliography with 6 fully cited sources:
                
                FOUNDATIONAL SOURCES (1-2)
                [Full citation]
                - Key quotes: "..."
                - Impact metrics: X citations
                - Relevance: Brief explanation
                
                PRIMARY SOURCES (3-4)
                [Full citation]
                - Archive/Location: Specific detail
                - Key content: "..."
                - Authority: Source validation
                
                CONTEMPORARY SOURCES (5-6)
                [Full citation]
                - Currency: Publication date/timeframe
                - Key findings: "..."
                - Scholarly impact: Recent citation count"""
        )

        return [research_analyst, policy_media_analyst, source_curator], [
            research_analysis,
            policy_media_analysis,
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
    st.session_state.research_status = "🔄 Conducting research analysis..."
    st.session_state.policy_media_status = "⏳ Waiting to start policy and media analysis..."
    st.session_state.sources_status = "⏳ Waiting to start source curation..."
    
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Create and run crew
            logger.info(f"Creating agents and tasks (Attempt {retry_count + 1}/{max_retries})")
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
            
            logger.info(f"Starting crew kickoff (Attempt {retry_count + 1}/{max_retries})")
            output = crew.kickoff()
            
            if output and str(output).strip():
                logger.info("Successfully received output from crew")
                output_str = str(output)
                
                # Process output
                sections = output_str.split("# Agent:")
                if len(sections) <= 1:
                    raise ValueError("Invalid output format from research agents")
                    
                for section in sections[1:]:
                    if "Research Analyst" in section:
                        st.session_state.research_status = "✅ Research Analysis Complete"
                        st.session_state.policy_media_status = "🔄 Analyzing policy and media..."
                    elif "Policy & Media Analyst" in section:
                        st.session_state.policy_media_status = "✅ Policy & Media Analysis Complete"
                        st.session_state.sources_status = "🔄 Curating sources..."
                    elif "Source Curator" in section:
                        st.session_state.sources_status = "✅ Source Curation Complete"
                
                st.session_state.research_results = output_str
                break
            
            retry_count += 1
            if retry_count < max_retries:
                time.sleep(2 * retry_count)
                
        except Exception as e:
            logger.error(f"Attempt {retry_count + 1} failed: {str(e)}")
            retry_count += 1
            if retry_count == max_retries:
                error_msg = f"Research process failed: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
                st.session_state.research_status = "❌ Research Analysis Failed"
                st.session_state.policy_media_status = "❌ Policy & Media Analysis Failed"
                st.session_state.sources_status = "❌ Source Curation Failed"
            time.sleep(2 * retry_count)
            
    st.session_state.is_processing = False

def main():
    st.title("📚 South Asian History Research Lite")
    st.write("Analyze historical topics in South Asian history with streamlined AI research agents!")

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
    st.header("1️⃣ Research Analysis")
    st.write(st.session_state.research_status)
    
    st.header("2️⃣ Policy & Media Analysis")
    st.write(st.session_state.policy_media_status)
    
    st.header("3️⃣ Sources & Bibliography")
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
        - Historical and cultural analysis
        - Policy impact and media representation analysis
        - Curated academic sources with citations
    """)

if __name__ == "__main__":
    main()
