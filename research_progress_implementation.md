# Research Progress Implementation Pseudocode

## 1. Cache Management

```python
# Cache research progress and results
@st.cache_data
def cache_research_state(session_id: str, state_data: dict):
    """
    Cache the current state of research including:
    - Overall progress
    - Agent activities
    - Intermediate results
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'state': state_data
    }

@st.cache_data
def cache_agent_activity(session_id: str, agent_name: str, activity: dict):
    """
    Cache individual agent activities:
    - Current task
    - Progress
    - Thoughts
    - Tool usage
    """
    return {
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'agent': agent_name,
        'activity': activity
    }

@st.cache_data
def get_cached_research_state(session_id: str):
    """Retrieve cached research state"""
    # Return None if no cache exists
    pass
```

## 2. Progress Tracking Components

```python
def initialize_research_interface():
    """Set up the main research interface with progress components"""
    
    # Main status container
    with st.status("Research in Progress", expanded=True) as status:
        # Phase progress indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            research_progress = st.progress(0, "Research Analysis")
        with col2:
            policy_progress = st.progress(0, "Policy Analysis")
        with col3:
            sources_progress = st.progress(0, "Source Curation")
            
        # Activity log container
        with st.container(height=300, border=True):
            st.subheader("🔄 Activity Log")
            activity_placeholder = st.empty()
            
        # Debug information (collapsible)
        with st.expander("🔍 Debug Information"):
            debug_placeholder = st.empty()

def update_research_status(status_container, phase: str, progress: float, message: str):
    """Update status for specific research phase"""
    if phase == "research":
        status_container.progress(progress, f"Research Analysis: {message}")
    elif phase == "policy":
        status_container.progress(progress, f"Policy Analysis: {message}")
    elif phase == "sources":
        status_container.progress(progress, f"Source Curation: {message}")
```

## 3. Enhanced Progress Callback

```python
class EnhancedProgressCallback:
    """Enhanced callback handler for CrewAI with Streamlit integration"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = time.time()
        
    def on_tool_start(self, agent_name: str, tool_name: str, input_data: str):
        """Handle tool start event"""
        activity = {
            'type': 'tool_start',
            'tool': tool_name,
            'input': input_data
        }
        cached_activity = cache_agent_activity(self.session_id, agent_name, activity)
        self._update_activity_log(cached_activity)
        
    def on_tool_end(self, agent_name: str, tool_name: str, output: str):
        """Handle tool completion event"""
        activity = {
            'type': 'tool_end',
            'tool': tool_name,
            'output_summary': summarize_output(output)
        }
        cached_activity = cache_agent_activity(self.session_id, agent_name, activity)
        self._update_activity_log(cached_activity)
        
    def on_agent_start(self, agent_name: str):
        """Handle agent start event"""
        activity = {
            'type': 'agent_start',
            'estimated_time': self._calculate_estimated_time()
        }
        cached_activity = cache_agent_activity(self.session_id, agent_name, activity)
        self._update_activity_log(cached_activity)
        
    def on_agent_end(self, agent_name: str):
        """Handle agent completion event"""
        activity = {
            'type': 'agent_end',
            'duration': time.time() - self.start_time
        }
        cached_activity = cache_agent_activity(self.session_id, agent_name, activity)
        self._update_activity_log(cached_activity)
        
    def _update_activity_log(self, activity_data: dict):
        """Update activity log in Streamlit interface"""
        # Format and display activity in the log container
        pass
        
    def _calculate_estimated_time(self) -> float:
        """Calculate estimated remaining time based on previous runs"""
        # Use cached data to estimate completion time
        pass
```

## 4. Error Handling

```python
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
            with st.expander("Error Details"):
                st.json(error.details)
    
    # Cache error state
    cache_research_state(
        session_id=st.session_state.user_id,
        state_data={
            'error': {
                'phase': error.phase,
                'message': error.message,
                'details': error.details
            }
        }
    )
```

## 5. Main Research Flow

```python
def conduct_research(topic: str):
    """Main research execution flow with progress tracking"""
    
    try:
        # Initialize session state if needed
        if 'user_id' not in st.session_state:
            st.session_state.user_id = f"session_{uuid.uuid4()}"
        
        # Set up progress tracking interface
        status_container = initialize_research_interface()
        
        # Create callback handler
        callback_handler = EnhancedProgressCallback(st.session_state.user_id)
        
        # Create and configure agents with callback
        agents, tasks = create_agents_and_tasks(
            research_topic=topic,
            callback_handler=callback_handler
        )
        
        # Create crew with progress tracking
        crew = Crew(
            agents=agents,
            tasks=tasks,
            verbose=True,
            process=Process.sequential,
            callback_handler=callback_handler
        )
        
        # Execute research with progress updates
        with status_container:
            st.write("🔄 Starting research process...")
            
            # Start research execution
            result = crew.kickoff()
            
            # Cache final results
            cache_research_state(
                session_id=st.session_state.user_id,
                state_data={
                    'status': 'complete',
                    'result': result
                }
            )
            
            # Update status to complete
            status_container.update(
                label="Research Complete",
                state="complete",
                expanded=False
            )
            
            return result
            
    except Exception as e:
        error = ResearchError(
            message=str(e),
            phase="research",
            details={'traceback': traceback.format_exc()}
        )
        handle_research_error(error, status_container)
        raise
```

## 6. Implementation Steps

1. Add cache decorators and functions
2. Implement EnhancedProgressCallback class
3. Create status and progress UI components
4. Add error handling
5. Modify main research flow
6. Test and validate all components

## 7. Usage Example

```python
def main():
    st.title("📚 South Asian History Research Pro")
    
    # Research topic input
    research_topic = st.text_input(
        "Enter your research topic",
        placeholder="e.g., Cultural transformation in Assam"
    )
    
    # Start research button
    if st.button("Start Research", disabled=st.session_state.get('is_processing', False)):
        if research_topic:
            try:
                # Conduct research with progress tracking
                result = conduct_research(research_topic)
                
                # Display results in tabs
                if result:
                    tab1, tab2 = st.tabs(["📝 Results", "📊 Analysis"])
                    with tab1:
                        st.text_area("Research Results", result, height=300)
                    with tab2:
                        # Show analytics about the results
                        display_research_analytics(result)
                        
            except Exception as e:
                st.error(f"Research failed: {str(e)}")
        else:
            st.error("Please enter a research topic")
```

This implementation provides:
- Real-time progress tracking
- Detailed activity logging
- Error handling and recovery
- Result caching
- Analytics visualization

The next step would be to implement these components one by one, starting with the cache system and progress tracking interface.
