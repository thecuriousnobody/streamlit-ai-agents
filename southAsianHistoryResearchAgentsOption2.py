import streamlit as st
import os
from crewai import Agent, Task, Crew, Process, LLM
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_tools import search_api_tool, google_scholar_tool, news_archive_tool
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchAgent:
    def __init__(self, role, goal, backstory, tools, llm=None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.llm = llm
        
    def execute(self, task):
        try:
            logger.info(f"Agent {self.role} starting task: {task}")
            tool_outputs = []
            
            # Gather information using tools
            for tool in self.tools:
                logger.info(f"Using tool: {tool.name}")
                try:
                    query = f"{task} {self.goal}"
                    if tool.name == "Google Scholar Search":
                        results = tool.func(query, num_results=5)
                        formatted_result = self._format_scholar_results(results)
                    elif tool.name == "News Archive Search":
                        results = tool.func(query, start_year=1900)
                        formatted_result = self._format_news_results(results)
                    else:
                        formatted_result = tool.func(query)
                    
                    tool_outputs.append(f"\n{tool.name} Results:\n{formatted_result}")
                except Exception as e:
                    logger.error(f"Error with tool {tool.name}: {str(e)}")
                    tool_outputs.append(f"Error with {tool.name}: {str(e)}")
            
            # Use LLM to analyze and synthesize the results
            if self.llm:
                analysis = self._llm_analysis(task, tool_outputs)
            else:
                analysis = self._compile_analysis(task, tool_outputs)
                
            return analysis
            
        except Exception as e:
            logger.error(f"Error in execute: {str(e)}")
            return f"Error during research: {str(e)}"
    
    def _llm_analysis(self, task, tool_outputs):
        try:
            # Prepare the prompt for the LLM
            prompt = f"""As a {self.role} with the following background: {self.backstory}
            
            Goal: {self.goal}
            Task: {task}
            
            Based on the following research results, provide a comprehensive analysis:
            
            {chr(10).join(tool_outputs)}
            
            Please provide:
            1. A synthesis of the key findings
            2. Critical analysis of the sources
            3. Connections to broader historical context
            4. Areas that need further research
            """
            
            # Get LLM response
            response = self.llm.predict(prompt)
            
            # Format the response
            analysis = f"\n\nAnalysis from {self.role}:\n"
            analysis += f"Task: {task}\n"
            analysis += f"Background: {self.backstory}\n"
            analysis += f"Goal: {self.goal}\n\n"
            analysis += "Synthesized Findings:\n"
            analysis += response
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in LLM analysis: {str(e)}")
            return self._compile_analysis(task, tool_outputs)  # Fallback to basic analysis
    
    def _format_scholar_results(self, results):
        formatted = []
        for result in results:
            entry = f"""
Title: {result.get('title')}
Authors: {', '.join(result.get('authors', []))}
Year: {result.get('year', 'N/A')}
Citations: {result.get('citations', 'N/A')}
Snippet: {result.get('snippet', 'N/A')}
"""
            formatted.append(entry)
        return "\n".join(formatted)
    
    def _format_news_results(self, results):
        formatted = []
        for result in results:
            entry = f"""
Title: {result.get('title')}
Source: {result.get('source', 'N/A')}
Date: {result.get('date', 'N/A')}
Snippet: {result.get('snippet', 'N/A')}
"""
            formatted.append(entry)
        return "\n".join(formatted)
    
    def _compile_analysis(self, task, tool_outputs):
        analysis = f"\n\nAnalysis from {self.role}:\n"
        analysis += f"Task: {task}\n"
        analysis += f"Background: As a {self.backstory},\n"
        analysis += f"Goal: {self.goal}\n\n"
        analysis += "Findings:\n"
        analysis += "\n".join(tool_outputs)
        return analysis

def create_agents_and_tasks(research_topic):
    # Initialize LLM if API key is available
    try:
        ClaudeSonnet = LLM(
            api_key=st.secrets["ANTHROPIC_API_KEY"],
            model="claude-3-sonnet-20240229"
        )
    except Exception as e:
        logger.warning(f"Could not initialize LLM: {str(e)}")
        ClaudeSonnet = None

    # Create agents with specific tool combinations
    historical_analyst = Agent(
        role="Historical Analyst",
        goal=f"Analyze historical development and cultural transformation in {research_topic}",
        backstory="Expert in South Asian history specializing in cultural transformation and minority experiences",
        tools=[google_scholar_tool, search_api_tool],
        llm=ClaudeSonnet
    )

    media_analyzer = Agent(
        role="Media Content Analyzer",
        goal=f"Track and analyze media representations related to {research_topic}",
        backstory="Media studies expert focusing on minority representation and narrative analysis",
        tools=[search_api_tool, news_archive_tool],
        llm=ClaudeSonnet
    )

    academic_curator = Agent(
        role="Academic Source Curator",
        goal=f"Validate and synthesize academic sources on {research_topic}",
        backstory="""Former research librarian at Oxford's Bodleian Libraries, specializing in South Asian studies. 
        Developed innovative citation analysis methods to track scholarly discourse on minority communities. 
        Led digital humanities initiatives connecting historical archives across India, Bangladesh, and Pakistan. 
        Known for uncovering overlooked primary sources that challenge dominant historical narratives.""",
        tools=[google_scholar_tool, search_api_tool],
        llm=ClaudeSonnet
    )

    tasks = [
        Task(
            description=f"Conduct a comprehensive historical analysis of {research_topic}, focusing on cultural developments and societal changes",
            agent=historical_analyst,
            expected_output="Detailed historical analysis with verified academic sources"
        ),
        Task(
            description=f"Analyze media representation patterns and narrative evolution regarding {research_topic} across different time periods",
            agent=media_analyzer,
            expected_output="Media analysis report with source documentation"
        ),
        Task(
            description=f"Compile and validate academic sources supporting the analysis of {research_topic}, with focus on peer-reviewed research",
            agent=academic_curator,
            expected_output="Annotated bibliography with citation metrics"
        )
    ]

    return [historical_analyst, media_analyzer, academic_curator], tasks

def run_research(research_topic, progress_containers):
    agents, tasks = create_agents_and_tasks(research_topic)
    
    # Initialize all containers with waiting state
    progress_containers["historical"].info("⏳ Waiting to start historical analysis...")
    progress_containers["media"].info("⏳ Waiting to start media analysis...")
    progress_containers["sources"].info("⏳ Waiting to start source curation...")
    
    crew = Crew(agents=agents, tasks=tasks)
    
    try:
        output = crew.kickoff()
        return process_output(output, progress_containers)
    except Exception as e:
        logger.error(f"Error in run_research: {str(e)}")
        st.error(f"An error occurred during research: {str(e)}")
        return None

def process_output(output, progress_containers):
    sections = output.split("# Agent:")
    
    for section in sections[1:]:
        if "Historical Analyst" in section:
            progress_containers["historical"].success("✅ Historical Analysis Complete")
            progress_containers["historical"].markdown(section.strip())
        elif "Media Content Analyzer" in section:
            progress_containers["media"].success("✅ Media Analysis Complete")
            progress_containers["media"].markdown(section.strip())
        elif "Academic Source Curator" in section:
            progress_containers["sources"].success("✅ Source Curation Complete")
            progress_containers["sources"].markdown(section.strip())
    
    return output

# Streamlit interface setup
st.set_page_config(
    page_title="South Asian History Research",
    page_icon="📚",
    layout="wide"
)

# Check for API key
if "SEARCH_API_KEY" not in st.secrets:
    st.error("""
        Please set up your Search API key in Streamlit Cloud:
        1. Go to your app settings in Streamlit Cloud
        2. Navigate to the Secrets section
        3. Add the following secret:
        ```toml
        SEARCH_API_KEY = "your-searchapi-key"
        ```
    """)
    st.stop()

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
        progress_containers = {
            "historical": st.container(),
            "media": st.container(),
            "sources": st.container()
        }
        
        with progress_containers["historical"]:
            st.subheader("1️⃣ Historical Analysis")
        with progress_containers["media"]:
            st.subheader("2️⃣ Media Analysis")
        with progress_containers["sources"]:
            st.subheader("3️⃣ Academic Sources")
        
        st.write(f"Researching topic: {research_topic}")
        
        result = run_research(research_topic, progress_containers)
        
        if result:
            st.success("Research completed!")
            
            st.download_button(
                label="Download Research Results",
                data=result,
                file_name=f"research_results_{research_topic.replace(' ', '_')}.txt",
                mime="text/plain"
            )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Error in main app: {str(e)}")
        st.write("Please try again with a different topic or contact support if the issue persists.")

# Footer
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
