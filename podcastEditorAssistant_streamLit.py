import streamlit as st
import os
import time
from crewai import Agent, Task, Crew, Process, LLM
import sys

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
        
        # # If not in environment, try Streamlit secrets
        # if not value and hasattr(st, 'secrets'):
        #     value = st.secrets.get(key)
            
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
        
        Please set these variables in your Streamlit Cloud:
        1. Go to your app settings in Streamlit Cloud
        2. Navigate to the Secrets section
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

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = ""
if 'context_status' not in st.session_state:
    st.session_state.context_status = "⏳ Waiting to start context analysis..."
if 'visual_status' not in st.session_state:
    st.session_state.visual_status = "⏳ Waiting to start visual enhancement analysis..."
if 'content_status' not in st.session_state:
    st.session_state.content_status = "⏳ Waiting to start content research..."
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = ""
if 'current_chunk' not in st.session_state:
    st.session_state.current_chunk = 0
if 'total_chunks' not in st.session_state:
    st.session_state.total_chunks = 0

def split_transcript_into_chunks(transcript, chunk_size=4):
    """Split transcript into chunks of roughly chunk_size paragraphs each."""
    lines = transcript.split('\n')
    chunks = []
    current_chunk = []
    paragraph_count = 0
    
    for line in lines:
        current_chunk.append(line)
        
        # Count empty lines as paragraph separators
        if not line.strip():
            paragraph_count += 1
            
            # When we reach chunk_size paragraphs, save the chunk
            if paragraph_count >= chunk_size:
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                paragraph_count = 0
    
    # Add any remaining lines as the last chunk
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks

def create_agents_and_tasks(transcript_chunk, chunk_number, total_chunks):
    """Create podcast editing agents and tasks for a specific chunk."""
    
    context_analyzer = Agent(
        role="Context Analyzer",
        goal="""Analyze podcast transcript chunks to understand the conversation context
                and identify key discussion points that could benefit from visual enhancement""",
        backstory="""Expert content analyst specializing in understanding conversation flow
                    and context. Skilled at identifying key themes, important references,
                    and moments that could benefit from visual support.""",
        tools=[serper_search_tool],
        llm=ClaudeSonnet
    )

    visual_enhancement_specialist = Agent(
        role="Visual Enhancement Specialist",
        goal="""Based on conversation context, suggest specific visual enhancements
                including B-roll, screenshots, diagrams, and relevant imagery""",
        backstory="""Expert video editor specializing in visual storytelling and content enhancement.
                    Skilled at identifying moments where visual elements can strengthen the narrative
                    and improve viewer engagement.""",
        tools=[serper_search_tool],
        llm=ClaudeSonnet
    )

    content_researcher = Agent(
        role="Content Researcher",
        goal="""Research and find specific supplementary content that matches the
                conversation context and enhances viewer understanding""",
        backstory="""Skilled researcher with expertise in finding high-quality supplementary
                    content from various sources. Experienced in identifying credible visual
                    and academic resources that can enhance video content.""",
        tools=[serper_search_tool, serper_scholar_tool],
        llm=ClaudeHaiku
    )

    context_analysis = Task(
        description=f"""Analyze this chunk ({chunk_number}/{total_chunks}) of the podcast transcript:

            1. Conversation Context
                - Main topics being discussed
                - Key points or arguments made
                - Important references or examples mentioned
            2. Visual Enhancement Opportunities
                - Moments that need visual clarification
                - References that could benefit from visual aids
                - Complex concepts that need illustration
            
            Transcript chunk to analyze:
            {transcript_chunk}""",
        agent=context_analyzer,
        expected_output="""Context analysis report with:
            1. MAIN DISCUSSION POINTS
               - [Timestamp] Topic summary
               - Key arguments/points
               - Important references
            
            2. VISUAL OPPORTUNITIES
               - [Timestamp] Moments needing visuals
               - Concepts requiring illustration
               - Reference visualization needs"""
    )

    visual_enhancement = Task(
        description=f"""Based on the conversation context in this chunk ({chunk_number}/{total_chunks}),
            suggest specific visual enhancements:
            1. B-roll Opportunities
                - Identify moments where B-roll footage could enhance the narrative
                - Suggest specific types of B-roll footage
                - Provide timestamps for each suggestion
            2. Visual Aid Recommendations
                - Points where diagrams/charts could clarify concepts
                - Opportunities for showing screenshots or images
                - Moments where on-screen text could reinforce key points
            
            Transcript chunk to analyze:
            {transcript_chunk}""",
        agent=visual_enhancement_specialist,
        expected_output="""Visual enhancement suggestions with:
            1. B-ROLL NEEDS
               - [Timestamp] Description of needed B-roll
               - Specific visual elements to include
               - Suggested duration and style
            
            2. VISUAL AIDS
               - [Timestamp] Type of visual aid needed
               - Description of content to show
               - Purpose and impact"""
    )

    content_research = Task(
        description=f"""Research and identify specific supplementary content for this chunk ({chunk_number}/{total_chunks}):
            1. Find relevant visual content
                - Images that illustrate key points
                - Video clips that demonstrate concepts
                - Websites or resources to showcase
            2. Provide specific URLs and sources
                - Direct links to suggested content
                - Usage rights/licensing information
                - Alternative options if primary suggestions unavailable
            
            Transcript chunk to analyze:
            {transcript_chunk}""",
        agent=content_researcher,
        expected_output="""Content resource list with:
            1. VISUAL RESOURCES
               - [Timestamp] URLs to specific images/videos
               - Source and licensing information
               - Alternative options
            
            2. ADDITIONAL RESOURCES
               - [Timestamp] Supplementary websites
               - Background information sources
               - Further reading suggestions"""
    )

    return [context_analyzer, visual_enhancement_specialist, content_researcher], [
        context_analysis,
        visual_enhancement,
        content_research
    ]

def analyze_podcast_chunk(transcript_chunk, chunk_number, total_chunks):
    """Analyze a single chunk of the podcast transcript."""
    try:
        agents, tasks = create_agents_and_tasks(transcript_chunk, chunk_number, total_chunks)
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential,
            memory=False,
            max_rpm=30
        )
        
        result = crew.kickoff()
        return result
    except Exception as e:
        st.error(f"Error analyzing chunk {chunk_number}: {str(e)}")
        return None

def analyze_podcast(transcript):
    """Run the podcast analysis process on chunks of the transcript."""
    chunks = split_transcript_into_chunks(transcript)
    st.session_state.total_chunks = len(chunks)
    
    st.write(f"\nSplit transcript into {st.session_state.total_chunks} chunks for analysis")
    all_results = []
    
    for i, chunk in enumerate(chunks, 1):
        st.session_state.current_chunk = i
        st.session_state.context_status = f"🔄 Analyzing context in chunk {i}/{st.session_state.total_chunks}..."
        st.session_state.visual_status = "⏳ Waiting for visual enhancement analysis..."
        st.session_state.content_status = "⏳ Waiting for content research..."
        
        result = analyze_podcast_chunk(chunk, i, st.session_state.total_chunks)
        
        if result:
            # Update status based on result sections
            if "Context analysis report" in str(result):
                st.session_state.context_status = f"✅ Context analysis complete for chunk {i}"
                st.session_state.visual_status = f"🔄 Analyzing visual enhancements for chunk {i}..."
            if "Visual enhancement suggestions" in str(result):
                st.session_state.visual_status = f"✅ Visual enhancement complete for chunk {i}"
                st.session_state.content_status = f"🔄 Researching content for chunk {i}..."
            if "Content resource list" in str(result):
                st.session_state.content_status = f"✅ Content research complete for chunk {i}"
            
            all_results.append(f"\n\nCHUNK {i}/{st.session_state.total_chunks} ANALYSIS:\n{result}")
    
    # Mark all complete when done
    st.session_state.context_status = "✅ Context Analysis Complete"
    st.session_state.visual_status = "✅ Visual Enhancement Analysis Complete"
    st.session_state.content_status = "✅ Content Research Complete"
    
    return "\n".join(all_results) if all_results else None

def main():
    st.title("🎥 Podcast Visual Enhancement Assistant")
    st.write("Analyze podcast transcripts and get AI-powered suggestions for visual enhancements!")

    # File uploader for transcript
    uploaded_file = st.file_uploader("Upload your podcast transcript", type=['txt'])
    
    # Start analysis button
    if st.button("Start Analysis", disabled=st.session_state.is_processing):
        if uploaded_file is None:
            st.error("Please upload a transcript file")
            return
            
        st.session_state.is_processing = True
        st.session_state.error_message = ""
        
        # Read and analyze transcript
        transcript = uploaded_file.getvalue().decode()
        result = analyze_podcast(transcript)
        
        if result:
            st.session_state.analysis_results = result
        
        st.session_state.is_processing = False

    # Display error message if any
    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    st.divider()

    # Display progress
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.header("1️⃣ Context Analysis")
        st.write(st.session_state.context_status)
    
    with col2:
        st.header("2️⃣ Visual Enhancement")
        st.write(st.session_state.visual_status)
    
    with col3:
        st.header("3️⃣ Content Research")
        st.write(st.session_state.content_status)

    # Display results
    if st.session_state.analysis_results:
        st.text_area("Analysis Results", st.session_state.analysis_results, height=400)
        
        # Download button
        if st.download_button(
            "Download Results",
            data=st.session_state.analysis_results,
            file_name="podcast_visual_enhancement_suggestions.txt",
            mime="text/plain"
        ):
            st.success("File downloaded successfully!")

    st.divider()
    
    # How to use section
    st.markdown("""
        ### How to use:
        1. Upload your podcast transcript file (txt format)
        2. Click 'Start Analysis' to begin the enhancement analysis
        3. Watch as the AI analyzes your content chunk by chunk
        4. Download the comprehensive enhancement suggestions

        The AI will provide:
        - Context analysis of conversation segments
        - B-roll and visual aid suggestions with timestamps
        - Relevant supplementary content with URLs
    """)

if __name__ == "__main__":
    main()
