import os
import logging
from crewai import Agent, Task, Crew, Process, LLM
from search_tools import (
    search_api_tool,
    google_scholar_tool,
    news_archive_tool,
    local_archives_tool,
    legal_database_tool
)

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

# Initialize LLM instances
try:
    ClaudeSonnet = LLM(
        model="claude-3-sonnet-20240229",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=8192,
        temperature=0.6
    )
    ClaudeHaiku = LLM(
        model="claude-3-haiku-20240229",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=8192,
        temperature=0.6
    )
except Exception as e:
    logger.warning(f"Could not initialize LLM: {str(e)}")
    ClaudeSonnet = None
    ClaudeHaiku = None

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
            tools= [search_api_tool],
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
                [Each with specific dates and significance]
                
                CULTURAL TRANSFORMATIONS (3-4)
                [Each with concrete examples and impact]
                
                COMMUNITY EXPERIENCES (5-6)
                [Each with evidence and perspectives]
                
                Each finding should be precisely dated and referenced."""
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
                [Each with implementation details and community impact]
                
                MEDIA NARRATIVES (3-4)
                [Each with specific examples and evolution over time]
                
                PUBLIC DISCOURSE (5-6)
                [Each with evidence and analysis]
                
                Each finding must include dates and specific supporting evidence."""
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
                
                Total response not to exceed 1000 words.""",
            agent=source_curator,
            expected_output="""Structured bibliography with 6 fully cited sources:
                
                FOUNDATIONAL SOURCES (1-2)
                [Full citations with Chicago style]
                - Key quotes: "..."
                - Impact metrics: X citations
                - Relevance: Brief explanation
                
                PRIMARY SOURCES (3-4)
                [Full citations]
                - Archive/Location: Specific detail
                - Key content: "..."
                - Authority: Source validation
                
                CONTEMPORARY SOURCES (5-6)
                [Full citations]
                - Currency: Publication date/timeframe
                - Key findings: "..."
                - Scholarly impact: Recent citation count
                
                Each source must include all required elements for academic citation."""
        )

        return [research_analyst, policy_media_analyst, source_curator], [
            research_analysis,
            policy_media_analysis,
            source_curation
        ]
    except Exception as e:
        logger.error(f"Error creating agents: {str(e)}")
        return None, None

def conduct_research(research_topic):
    """Main research function"""
    logger.info(f"Starting research on: {research_topic}")
    
    agents, tasks = create_agents_and_tasks(research_topic)
    if not agents or not tasks:
        logger.error("Failed to create agents and tasks")
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
        result = crew.kickoff()
        
        # Save results to file
        output_dir = "research_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        filename = f"{research_topic.replace(' ', '_')}_research.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Research Results for: {research_topic}\n\n")
            f.write(str(result))
        
        logger.info(f"Research saved to: {filepath}")
        return result
        
    except Exception as e:
        logger.error(f"Research process failed: {str(e)}")
        return None

if __name__ == "__main__":
    topic = input("Enter your South Asian history research topic: ")
    results = conduct_research(topic)
    if results:
        print("\nResearch Results:")
        print(results)
    else:
        print("Research failed. Check the logs for details.")
