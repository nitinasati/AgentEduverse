import streamlit as st
import asyncio
import aiohttp
import json
import os
from datetime import datetime
from typing import Dict, Any, List
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger

# Configure page
st.set_page_config(
    page_title="AgentEduverse",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-buy {
        color: #28a745;
        font-weight: bold;
    }
    .recommendation-sell {
        color: #dc3545;
        font-weight: bold;
    }
    .recommendation-hold {
        color: #ffc107;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .system-message {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

class AgentEduverseClient:
    """Client for interacting with the AgentEduverse system"""
    
    def __init__(self):
        self.master_agent_url = os.getenv('MASTER_AGENT_URL', 'http://localhost:8001')
        self.stock_agent_url = os.getenv('STOCK_AGENT_URL', 'http://localhost:8002')
        self.mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8003')
        self.logger = logger.get_logger("frontend_client")
    
    async def send_message_to_master(self, message: str, session_id: str = None) -> Dict[str, Any]:
        """Send message to master agent"""
        try:
            self.logger.info(f"Sending message to master agent: {message[:100]}...", extra_data={
                'message_length': len(message),
                'master_agent_url': self.master_agent_url,
                'session_id': session_id,
                'step': 'message_send_start'
            })
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.master_agent_url}/process_task",
                    json={
                        "task": message,
                        "session_id": session_id
                    }
                ) as response:
                    self.logger.info(f"Master agent response status: {response.status}", extra_data={
                        'response_status': response.status,
                        'session_id': session_id,
                        'step': 'response_received'
                    })
                    
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Master agent response received: {str(result)[:200]}...", extra_data={
                            'response_length': len(str(result)),
                            'session_id': session_id,
                            'step': 'response_success'
                        })
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Master agent error: {error_text}", extra_data={
                            'response_status': response.status,
                            'error_text': error_text,
                            'session_id': session_id,
                            'step': 'response_error'
                        })
                        return {"error": f"Master agent error: {error_text}"}
        except Exception as e:
            self.logger.error(f"Connection error: {str(e)}", extra_data={
                'error': str(e),
                'error_type': type(e).__name__,
                'session_id': session_id,
                'step': 'connection_error'
            })
            return {"error": f"Connection error: {str(e)}"}
    
    async def analyze_stock(self, symbol: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Analyze stock using stock agent"""
        try:
            self.logger.info(f"Analyzing stock: {symbol} with type: {analysis_type}", extra_data={
                'symbol': symbol,
                'analysis_type': analysis_type,
                'stock_agent_url': self.stock_agent_url,
                'step': 'stock_analysis_start'
            })
            
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self.stock_agent_url}/analyze_stock",
                    json={
                        "symbol": symbol,
                        "analysis_type": analysis_type
                    }
                ) as response:
                    self.logger.info(f"Stock agent response status: {response.status}", extra_data={
                        'response_status': response.status,
                        'symbol': symbol,
                        'step': 'stock_response_received'
                    })
                    
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"Stock agent response received: {str(result)[:200]}...", extra_data={
                            'response_length': len(str(result)),
                            'symbol': symbol,
                            'step': 'stock_response_success'
                        })
                        return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Stock agent error: {error_text}", extra_data={
                            'response_status': response.status,
                            'error_text': error_text,
                            'symbol': symbol,
                            'step': 'stock_response_error'
                        })
                        return {"error": f"Stock agent error: {error_text}"}
        except Exception as e:
            self.logger.error(f"Stock analysis connection error: {str(e)}", extra_data={
                'error': str(e),
                'error_type': type(e).__name__,
                'symbol': symbol,
                'step': 'stock_connection_error'
            })
            return {"error": f"Connection error: {str(e)}"}
    
    async def get_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.master_agent_url}/agents") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": "Failed to get agents status"}
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}
    
    async def get_mcp_tools(self) -> Dict[str, Any]:
        """Get available MCP tools"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mcp_server_url}/tools") as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        return {"error": "Failed to get MCP tools"}
        except Exception as e:
            return {"error": f"Connection error: {str(e)}"}

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if 'client' not in st.session_state:
        st.session_state.client = AgentEduverseClient()

def display_header():
    """Display the main header"""
    st.markdown('<h1 class="main-header">🤖 AgentEduverse</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
            Powered by AI Agents • Real-time Stock Analysis • Investment Recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar with system information"""
    with st.sidebar:
        st.header("🔧 System Status")
        
        # Check system health
        if st.button("🔄 Check System Health"):
            with st.spinner("Checking system health..."):
                health_status = asyncio.run(check_system_health())
                st.json(health_status)
        
        st.header("📊 Quick Actions")
        
        # Quick stock analysis
        symbol = st.text_input("Stock Symbol", placeholder="e.g., AAPL, MSFT, GOOGL")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["comprehensive", "technical", "financial", "recommendation"]
        )
        
        if st.button("📈 Analyze Stock"):
            if symbol:
                with st.spinner(f"Analyzing {symbol.upper()}..."):
                    result = asyncio.run(st.session_state.client.analyze_stock(symbol, analysis_type))
                    if "error" not in result:
                        st.success(f"Analysis completed for {symbol.upper()}")
                        st.text_area("Analysis Result", result.get("result", ""), height=300)
                    else:
                        st.error(f"Analysis failed: {result['error']}")
        
        st.header("🎯 Example Queries")
        example_queries = [
            "Analyze Apple stock and give me a buy/sell recommendation",
            "What's the technical analysis for Tesla?",
            "Compare Microsoft and Google stocks",
            "Show me the latest news sentiment for Amazon",
            "What are the best performing stocks in tech sector?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query[:20]}"):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()

async def check_system_health():
    """Check health of all system components"""
    health_status = {}
    
    # Check master agent
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{st.session_state.client.master_agent_url}/health") as response:
                health_status["master_agent"] = await response.json() if response.status == 200 else {"status": "unhealthy"}
    except Exception:
        health_status["master_agent"] = {"status": "unreachable"}
    
    # Check stock agent
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{st.session_state.client.stock_agent_url}/health") as response:
                health_status["stock_agent"] = await response.json() if response.status == 200 else {"status": "unhealthy"}
    except Exception:
        health_status["stock_agent"] = {"status": "unreachable"}
    
    # Check MCP server
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{st.session_state.client.mcp_server_url}/health") as response:
                health_status["mcp_server"] = await response.json() if response.status == 200 else {"status": "unhealthy"}
    except Exception:
        health_status["mcp_server"] = {"status": "unreachable"}
    
    return health_status

def display_chat_interface():
    """Display the main chat interface"""
    st.header("💬 Chat with AI Agents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about stocks, get analysis, or request recommendations..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from master agent
        with st.chat_message("assistant"):
            with st.spinner("🤖 Processing your request..."):
                # Get logger for frontend
                frontend_logger = logger.get_logger("frontend_interface")
                
                frontend_logger.info("Processing user request: " + prompt[:100] + "...", extra_data={
                    'prompt_length': len(prompt),
                    'session_id': st.session_state.session_id,
                    'step': 'user_request_processing'
                })
                
                response = asyncio.run(
                    st.session_state.client.send_message_to_master(
                        prompt, 
                        st.session_state.session_id
                    )
                )
                
                frontend_logger.info(f"Response received: {str(response)[:200]}...", extra_data={
                    'response_length': len(str(response)),
                    'session_id': st.session_state.session_id,
                    'step': 'response_processing'
                })
                
                if "error" not in response:
                    assistant_response = response.get("response", "I apologize, but I couldn't process your request.")
                    frontend_logger.info(f"Assistant response: {assistant_response[:200]}...", extra_data={
                        'response_length': len(assistant_response),
                        'session_id': st.session_state.session_id,
                        'step': 'assistant_response_success'
                    })
                    st.markdown(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                else:
                    error_message = f"❌ Error: {response['error']}"
                    frontend_logger.error(f"Error in response: {error_message}", extra_data={
                        'error': response['error'],
                        'session_id': st.session_state.session_id,
                        'step': 'assistant_response_error'
                    })
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

def display_agent_status():
    """Display agent status and capabilities"""
    st.header("🤖 Agent Status")
    
    if st.button("🔄 Refresh Agent Status"):
        with st.spinner("Getting agent status..."):
            agents_status = asyncio.run(st.session_state.client.get_agents_status())
            
            if "agents" in agents_status:
                for agent in agents_status["agents"]:
                    with st.expander(f"📋 {agent.get('name', 'Unknown Agent')}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Status:** {agent.get('status', 'Unknown')}")
                            st.write(f"**Description:** {agent.get('description', 'No description')}")
                        
                        with col2:
                            st.write(f"**Capabilities:** {', '.join(agent.get('capabilities', []))}")
                            st.write(f"**Last Seen:** {agent.get('last_seen', 'Unknown')}")
            else:
                st.error(f"Failed to get agent status: {agents_status.get('error', 'Unknown error')}")

def display_mcp_tools():
    """Display available MCP tools"""
    st.header("🔧 Available MCP Tools")
    
    if st.button("🔄 Refresh MCP Tools"):
        with st.spinner("Getting MCP tools..."):
            tools_status = asyncio.run(st.session_state.client.get_mcp_tools())
            
            if "tools" in tools_status:
                for tool in tools_status["tools"]:
                    with st.expander(f"🔧 {tool.get('name', 'Unknown Tool')}"):
                        st.write(f"**Description:** {tool.get('description', 'No description')}")
                        st.write("**Parameters:**")
                        st.json(tool.get('parameters', {}))
            else:
                st.error(f"Failed to get MCP tools: {tools_status.get('error', 'Unknown error')}")

def display_analytics_dashboard():
    """Display analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    # Create tabs for different analytics
    tab1, tab2, tab3 = st.tabs(["📈 Stock Performance", "📊 Market Trends", "🎯 Recommendations"])
    
    with tab1:
        st.subheader("Stock Performance Overview")
        # Placeholder for stock performance charts
        st.info("Stock performance charts will be displayed here based on user queries.")
    
    with tab2:
        st.subheader("Market Trends Analysis")
        # Placeholder for market trends
        st.info("Market trends and sentiment analysis will be displayed here.")
    
    with tab3:
        st.subheader("Investment Recommendations")
        # Placeholder for recommendations
        st.info("Recent investment recommendations will be displayed here.")

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat Interface", 
        "🤖 Agent Status", 
        "🔧 MCP Tools", 
        "📊 Analytics"
    ])
    
    with tab1:
        display_chat_interface()
    
    with tab2:
        display_agent_status()
    
    with tab3:
        display_mcp_tools()
    
    with tab4:
        display_analytics_dashboard()
    
    # Display sidebar
    display_sidebar()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem;">
        <p>AgentEduverse • Powered by LangChain, LangGraph, and AWS Bedrock</p>
        <p>Session ID: {}</p>
    </div>
    """.format(st.session_state.session_id), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
