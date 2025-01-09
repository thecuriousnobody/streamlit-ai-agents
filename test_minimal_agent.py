import os
import sys
import streamlit as st
from crewai import Agent, Task, Crew, Process, LLM
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search_tools import search_api_tool, google_scholar_tool, news_archive_tool
from pydantic import BaseModel
from typing import List, Dict

# class ResearchSource(BaseModel):
#     title: str
#     authors: List[str]
#     year: int
#     url: str
#     citation: str
#     snippet: str
#     relevance_score: float

# class SelectedSource(BaseModel):
#     citation: str
#     url: str
#     doi: str
#     key_quotes: List[str]
#     impact_metrics: Dict[str, any]
#     relevance: str

class ResearchAgent:
    def __init__(self, role, goal, backstory, tools, llm=None):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.llm = llm
        
    def execute(self, task):
        try:
            tool_outputs = []
            
            # Progress bar for tool execution
            progress_bar = st.progress(0)
            total_tools = len(self.tools)
            
            # Gather information using tools
            for idx, tool in enumerate(self.tools):
                try:
                    with st.spinner(f'Using {tool.name}...'):
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
                        
                        # Update progress
                        progress = (idx + 1) / total_tools
                        progress_bar.progress(progress)
                        
                except Exception as e:
                    tool_outputs.append(f"Error with {tool.name}: {str(e)}")
            
            # Use LLM to analyze and synthesize the results
            with st.spinner('Analyzing results...'):
                if self.llm:
                    analysis = self._llm_analysis(task, tool_outputs)
                else:
                    analysis = self._compile_analysis(task, tool_outputs)
                    
                return analysis
            
        except Exception as e:
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
            
        except Exception:
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
        ClaudeHaiku = LLM(
            model="claude-3-5-haiku-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_tokens=8192,
            temperature=0.6
        )
    except Exception as e:
        st.error(f"Could not initialize LLM: {str(e)}")
        return None, None

    # Create agents with specific tool combinations
    research_analyst = Agent(
        role="Research Analyst",
        goal="""Conduct comprehensive historical and ethnographic analysis of South Asian topics,
                focusing on cultural transformation, community experiences, and social dynamics""",
        backstory="""Expert historian and ethnographer specializing in South Asian studies with extensive 
                    fieldwork experience. Known for combining historical analysis with contemporary 
                    cultural understanding to provide deep insights into community dynamics and social transformation.""",
        tools=[google_scholar_tool, search_api_tool],
        llm=ClaudeHaiku
    )

    policy_media_analyst = Agent(
        role="Policy & Media Analyst",
        goal="""Analyze policy implementations and media representations, examining their impact
                on communities and public discourse""",
        backstory="""Experienced policy researcher and media analyst specializing in South Asian affairs.
                    Expert in analyzing government policies, legal frameworks, and their representation
                    in various media formats.""",
        tools=[search_api_tool, news_archive_tool],
        llm=ClaudeHaiku
    )

    source_curator = Agent(
        role="Source & Citation Specialist",
        goal="""Create a comprehensive research foundation by finding, validating, and properly citing key sources.""",
        backstory="""Research librarian and citation specialist with expertise in South Asian studies. 
                    Known for creating meticulous source documentation and finding precise supporting evidence.""",
        tools=[google_scholar_tool, search_api_tool],
        llm=ClaudeHaiku
    )

    # Task 1: Source Discovery and Logging
    source_discovery_desc = f"""Search and log ALL scholarly sources related to {research_topic}:
            
            For each discovered source, create a structured entry with:
            1. Full title and authors
            2. Publication year
            3. URL or DOI link
            4. Brief content snippet
            5. Initial relevance assessment
            
            Format each entry in clear markdown structure."""

    source_discovery = Task(
        description=source_discovery_desc,
        agent=source_curator,
        expected_output="""A comprehensive research log in markdown format:
            
            # Complete Research Log
            ## Source 1
            - Title: [title]
            - Authors: [authors]
            - Year: [year]
            - URL: [url]
            - Snippet: [relevant excerpt]
            - Initial Assessment: [relevance notes]
            
            [Repeat for all sources discovered]""",
        output_file="complete_research_log.md",
        async_execution=True
    )

    # Task 2: Source Curation with Full Context
    source_curation_desc = f"""From the research log, select and analyze the most relevant sources for {research_topic}:
            
            For each selected source provide:
            1. Full academic citation (Chicago style)
            2. Direct URL or DOI link
            3. 2-3 key quotes with page numbers
            4. Impact metrics and scholarly authority
            5. Source accessibility assessment
            
            Create two documentation sections:
            1. Selected Sources (6 total)
            2. Reference to Complete Research Log"""

    source_curation = Task(
        description=source_curation_desc,
        agent=source_curator,
        context=[source_discovery],
        expected_output="""# Curated Research Sources

            ## FOUNDATIONAL SOURCES
            1. [Full citation]
            - URL: [direct link]
            - Key Quotes: [quotes with page numbers]
            - Impact: [metrics]
            - Relevance: [assessment]
            
            [Repeat for other categories]
            
            ## RESEARCH LOG REFERENCE
            [Summary of broader sources considered]""",
        output_file="curated_sources.md"
    )

    # Task 3: Research Analysis
    research_analysis_desc = f"""Using the curated sources, analyze {research_topic} across these dimensions:
            
            1. Historical Context:
            - 2 major developments with dates
            - Academic interpretations of these events
            - Supporting evidence from sources
            
            2. Scholarly Discourse:
            - 2 key theoretical frameworks
            - Research methodologies used
            - Evidence of academic debate
            
            3. Current Understanding:
            - 2 contemporary scholarly perspectives
            - Emerging research directions
            - Gaps in current literature"""

    research_analysis = Task(
        description=research_analysis_desc,
        agent=research_analyst,
        context=[source_curation],
        expected_output="""Structured analysis linking to sources:
            
            HISTORICAL DEVELOPMENTS
            [Events + Academic interpretation + Source evidence]
            
            SCHOLARLY FRAMEWORKS
            [Theories + Methods + Debates]
            
            CURRENT PERSPECTIVES
            [Contemporary views + Research directions]""",
        output_file="research_analysis.md"
    )

    # Task 4: Policy and Media Analysis
    policy_media_desc = f"""Analyze policy and media dimensions of {research_topic} using scholarly sources:
            
            1. Policy Framework:
            - 2 key policy developments
            - Academic analysis of implementation
            - Evidence of impacts
            
            2. Media Representation:
            - 2 dominant scholarly interpretations
            - Research-based evidence
            - Methodological approaches
            
            3. Public Discourse:
            - 2 academic analyses of discourse
            - Research methodologies used
            - Scholarly debate points"""

    policy_media_analysis = Task(
        description=policy_media_desc,
        agent=policy_media_analyst,
        context=[source_curation, research_analysis],
        expected_output="""Integrated analysis with scholarly evidence:
            
            POLICY ANALYSIS
            [Developments + Academic interpretation + Evidence]
            
            MEDIA RESEARCH
            [Interpretations + Methods + Findings]
            
            DISCOURSE ANALYSIS
            [Academic perspectives + Research approaches]""",
        output_file="policy_media_analysis.md"
    )

    return [source_curator, research_analyst, policy_media_analyst], [
        source_discovery,  # First task - discovers and logs all sources
        source_curation,   # Second task - curates the best sources with full analysis
        research_analysis, # Third task - uses curated sources for research analysis
        policy_media_analysis  # Final task - integrates policy and media perspectives
    ]

def conduct_research(research_topic):
    """Main research function"""
    with st.spinner('Initializing research agents...'):
        agents, tasks = create_agents_and_tasks(research_topic)
        if not agents or not tasks:
            st.error("Failed to create agents and tasks")
            return None
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential,
            memory=False,
            max_rpm=30
        )
    
    try:
        with st.spinner('Conducting research...'):
            result = crew.kickoff()
            
            # Save results to file
            output_dir = "research_output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Create a smart, concise filename from the research topic
            # Extract first 3 significant words, remove common words
            words = research_topic.lower().split()
            common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            key_terms = [word for word in words if word not in common_words][:3]
            
            # Add timestamp for uniqueness
            from datetime import datetime
            timestamp = datetime.now().strftime("%y%m%d")
            
            # Combine terms with timestamp
            filename = f"research_{timestamp}_{'_'.join(key_terms)}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Research Results for: {research_topic}\n\n")
                f.write(str(result))
            
            return result, filename, filepath
        
    except Exception as e:
        st.error(f"Research process failed: {str(e)}")
        return None

def main():
    st.title("South Asian History Research Assistant")
    st.write("""
    This tool conducts comprehensive research on South Asian historical topics using multiple specialized agents:
    - Research Analyst: Historical and ethnographic analysis
    - Policy & Media Analyst: Policy implementation and media representation analysis
    - Source Curator: Source validation and citation management
    """)
    
    # Input section
    st.header("Research Topic")
    research_topic = st.text_input(
        "Enter your South Asian history research topic:",
        placeholder="e.g., The impact of partition on Bengali culture"
    )
    
    # Execute research when user submits
    if st.button("Conduct Research"):
        if not research_topic:
            st.warning("Please enter a research topic")
            return
            
        with st.spinner('Starting research process...'):
            results, filename, filepath = conduct_research(research_topic)
            
        if results and filename and filepath:
            # Display results in expandable sections
            st.header("Research Results")
            
            # Split results into sections
            sections = str(results).split("\n\nAnalysis from")
            
            for section in sections:
                if section.strip():
                    # Create a title for the section
                    title = section.split("\n")[0] if ":" in section else "Research Results"
                    with st.expander(title, expanded=True):
                        st.markdown(section)
            
            # Show file save location and download button
            st.success(f"Results saved to: research_output/{filename}")
            
            # Add download button
            with open(filepath, 'r', encoding='utf-8') as f:
                file_content = f.read()
                st.download_button(
                    label="📥 Download Research Results",
                    data=file_content,
                    file_name=filename,
                    mime="text/plain"
                )
        else:
            st.error("Research failed. Please check the error messages above.")

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        st.warning("python-dotenv not installed. Environment variables must be set manually.")
    
    main()
