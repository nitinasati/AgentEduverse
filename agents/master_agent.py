import asyncio
import json
import os
import aiohttp
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent
from utils.logger import logger

class AgentRegistry:
    """Registry for discovered agents"""
    
    def __init__(self):
        self.agents = {}
        self.logger = logger.get_logger("agent_registry")
    
    def register_agent(self, agent_info: Dict[str, Any]):
        """Register a new agent"""
        agent_id = agent_info.get('id', str(uuid.uuid4()))
        self.agents[agent_id] = {
            **agent_info,
            'registered_at': datetime.now().isoformat(),
            'last_seen': datetime.now().isoformat()
        }
        self.logger.info(f"Agent registered: {agent_info.get('name', 'Unknown')}", extra_data={
            'agent_id': agent_id,
            'agent_name': agent_info.get('name'),
            'capabilities': agent_info.get('capabilities', [])
        })
        return agent_id
    
    def update_agent_status(self, agent_id: str, status: str):
        """Update agent status"""
        if agent_id in self.agents:
            self.agents[agent_id]['status'] = status
            self.agents[agent_id]['last_seen'] = datetime.now().isoformat()
            self.logger.info(f"Agent status updated: {agent_id} -> {status}", extra_data={
                'agent_id': agent_id,
                'status': status
            })
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_capability(self, capability: str) -> List[Dict[str, Any]]:
        """Get agents with specific capability"""
        return [
            agent for agent in self.agents.values()
            if capability in agent.get('capabilities', [])
        ]
    
    def get_all_agents(self) -> List[Dict[str, Any]]:
        """Get all registered agents"""
        return list(self.agents.values())
    
    def remove_agent(self, agent_id: str):
        """Remove agent from registry"""
        if agent_id in self.agents:
            agent_name = self.agents[agent_id].get('name', 'Unknown')
            del self.agents[agent_id]
            self.logger.info(f"Agent removed: {agent_name}", extra_data={
                'agent_id': agent_id,
                'agent_name': agent_name
            })

class MasterAgent(BaseAgent):
    """Master agent that discovers and coordinates other agents"""
    
    def __init__(self):
        super().__init__(
            agent_name="MasterAgent",
            agent_description="Master agent responsible for discovering, coordinating, and routing tasks to specialized agents"
        )
        
        # Initialize agent registry
        self.registry = AgentRegistry()
        
        # Add master agent capabilities
        self.add_capability("agent_discovery")
        self.add_capability("task_routing")
        self.add_capability("coordination")
        self.add_capability("load_balancing")
        
        # Add tools for agent management
        self._add_agent_management_tools()
        
        # Task queue for coordination
        self.task_queue = asyncio.Queue()
        
        # Set scope boundaries for Master Agent
        self.set_scope_boundaries({
            'allowed_domains': [
                'agent coordination', 'task routing', 'system management', 
                'agent discovery', 'load balancing', 'health monitoring'
            ],
            'forbidden_actions': [
                'execute financial transactions', 'make investment decisions',
                'access sensitive data', 'modify system files', 'bypass security',
                'perform unauthorized actions', 'access user credentials'
            ],
            'max_iterations': 5,
            'timeout_seconds': 180,
            'rate_limits': {
                'requests_per_minute': 30,
                'requests_per_hour': 500
            }
        })
        
        # Add specific forbidden actions
        self.add_forbidden_action('execute stock trades')
        self.add_forbidden_action('provide financial advice')
        self.add_forbidden_action('access personal data')
        self.add_forbidden_action('modify agent code')
        
        self.logger.info("Master Agent initialized successfully", extra_data={
            'capabilities': self.capabilities,
            'tools_count': len(self.tools),
            'scope_boundaries': self.scope_boundaries
        })
    
    def _add_agent_management_tools(self):
        """Add tools for managing agents"""
        from langchain.tools import Tool
        
        # Tool to discover agents
        discover_tool = Tool(
            name="discover_agents",
            description="Discover and register new agents in the system",
            func=self._discover_agents
        )
        self.add_tool(discover_tool)
        
        # Tool to route tasks
        route_tool = Tool(
            name="route_task",
            description="Route a task to the most appropriate agent based on capabilities",
            func=self._route_task
        )
        self.add_tool(route_tool)
        
        # Tool to get agent status
        status_tool = Tool(
            name="get_agent_status",
            description="Get status and information about registered agents",
            func=self._get_agent_status
        )
        self.add_tool(status_tool)
        
        # Tool to coordinate multiple agents
        coordinate_tool = Tool(
            name="coordinate_agents",
            description="Coordinate multiple agents to work on a complex task",
            func=self._coordinate_agents
        )
        self.add_tool(coordinate_tool)
    
    async def _discover_agents(self, discovery_request: str = "") -> str:
        """Discover and register new agents"""
        try:
            self.logger.info("Starting agent discovery", extra_data={
                'discovery_request': discovery_request
            })
            
            # Simulate agent discovery (in real implementation, this would use A2A framework)
            discovered_agents = await self._scan_for_agents()
            
            # Register discovered agents
            registered_count = 0
            for agent_info in discovered_agents:
                self.registry.register_agent(agent_info)
                registered_count += 1
            
            result = f"Discovered and registered {registered_count} new agents"
            self.logger.info(f"Agent discovery completed: {result}", extra_data={
                'discovered_count': registered_count,
                'total_agents': len(self.registry.get_all_agents())
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Agent discovery failed: {str(e)}", extra_data={
                'error': str(e)
            })
            return f"Agent discovery failed: {str(e)}"
    
    async def _scan_for_agents(self) -> List[Dict[str, Any]]:
        """Scan for available agents (simulated A2A discovery)"""
        # In a real implementation, this would use Google A2A framework
        # For now, we'll simulate discovery of common agent types
        
        potential_agents = [
            {
                'id': str(uuid.uuid4()),
                'name': 'StockAgent',
                'description': 'Specialized agent for stock analysis and financial recommendations',
                'capabilities': ['stock_analysis', 'financial_data', 'investment_recommendations'],
                'endpoint': 'http://localhost:8002',
                'status': 'available'
            },
            {
                'id': str(uuid.uuid4()),
                'name': 'NewsAgent',
                'description': 'Agent for news analysis and sentiment processing',
                'capabilities': ['news_analysis', 'sentiment_analysis', 'market_news'],
                'endpoint': 'http://localhost:8004',
                'status': 'available'
            },
            {
                'id': str(uuid.uuid4()),
                'name': 'TechnicalAnalysisAgent',
                'description': 'Agent for technical analysis and chart patterns',
                'capabilities': ['technical_analysis', 'chart_patterns', 'indicators'],
                'endpoint': 'http://localhost:8005',
                'status': 'available'
            }
        ]
        
        # Simulate network discovery delay
        await asyncio.sleep(0.1)
        
        return potential_agents
    
    async def _route_task(self, task_description: str) -> str:
        """Route a task to the most appropriate agent"""
        try:
            self.logger.info(f"Routing task: {task_description}", extra_data={
                'task_description': task_description
            })
            
            # Analyze task to determine required capabilities
            required_capabilities = await self._analyze_task_requirements(task_description)
            
            # Find agents with matching capabilities
            suitable_agents = []
            for capability in required_capabilities:
                agents = self.registry.get_agents_by_capability(capability)
                suitable_agents.extend(agents)
            
            if not suitable_agents:
                return "No suitable agents found for this task"
            
            # Select the best agent (simple selection for now)
            selected_agent = suitable_agents[0]
            
            # Route the task
            result = await self._send_task_to_agent(selected_agent, task_description)
            
            self.logger.info("Task routed successfully", extra_data={
                'selected_agent': selected_agent.get('name'),
                'agent_id': selected_agent.get('id'),
                'required_capabilities': required_capabilities
            })
            
            return f"Task routed to {selected_agent.get('name')}: {result}"
            
        except Exception as e:
            self.logger.error(f"Task routing failed: {str(e)}", extra_data={
                'error': str(e),
                'task_description': task_description
            })
            return f"Task routing failed: {str(e)}"
    
    async def _analyze_task_requirements(self, task_description: str) -> List[str]:
        """Analyze task to determine required capabilities"""
        # Use LLM to analyze task requirements
        prompt = f"""
        Analyze the following task and determine what capabilities are required:
        
        Task: {task_description}
        
        Available capabilities:
        - stock_analysis: Analyzing stock prices, financial data, and market trends
        - financial_data: Processing financial statements, ratios, and metrics
        - investment_recommendations: Providing buy/sell/hold recommendations
        - news_analysis: Analyzing news articles and market sentiment
        - sentiment_analysis: Processing sentiment from text and news
        - market_news: Gathering and analyzing market-related news
        - technical_analysis: Analyzing price charts and technical indicators
        - chart_patterns: Identifying chart patterns and trends
        - indicators: Calculating and interpreting technical indicators
        
        Return only the capability names that are relevant to this task, separated by commas.
        """
        
        response = await self._get_llm_response(prompt, "task_analysis")
        
        # Parse capabilities from response
        capabilities = [cap.strip() for cap in response.split(',') if cap.strip()]
        
        return capabilities
    
    async def _send_task_to_agent(self, agent: Dict[str, Any], task: str) -> str:
        """Send task to a specific agent"""
        try:
            endpoint = agent.get('endpoint')
            if not endpoint:
                return f"Agent {agent.get('name')} has no endpoint configured"
            
            # For stock-related tasks, actually call the stock agent
            if 'stock_analysis' in agent.get('capabilities', []):
                return await self._call_stock_agent(task)
            
            # For other agents, simulate sending task
            await asyncio.sleep(0.1)  # Simulate network delay
            return f"Task sent to {agent.get('name')} successfully"
            
        except Exception as e:
            return f"Failed to send task to agent: {str(e)}"
    
    async def _call_stock_agent(self, task: str) -> str:
        """Call the stock agent for stock-related tasks"""
        try:
            import aiohttp
            import re
            
            # Extract stock symbol from task - look for common stock symbols
            common_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'AMD', 'INTC']
            symbol = None
            
            # First try to find exact matches for common symbols
            for common_symbol in common_symbols:
                if common_symbol in task.upper():
                    symbol = common_symbol
                    break
            
            # If no common symbol found, use regex to find any 1-5 letter uppercase sequence
            if not symbol:
                symbol_match = re.search(r'\b([A-Z]{1,5})\b', task.upper())
                if symbol_match:
                    symbol = symbol_match.group(1)
            
            if not symbol:
                return "No stock symbol found in the task. Please specify a stock symbol (e.g., AAPL, MSFT)."
            stock_agent_url = os.getenv('STOCK_AGENT_URL', 'http://localhost:8002')
            
            self.logger.info(f"Calling stock agent for symbol: {symbol}", extra_data={
                'symbol': symbol,
                'stock_agent_url': stock_agent_url,
                'task': task
            })
            
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{stock_agent_url}/analyze_stock",
                    json={
                        "symbol": symbol,
                        "analysis_type": "comprehensive"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get('result', 'Analysis completed but no result returned')
                    else:
                        error_text = await response.text()
                        return f"Stock agent error: {error_text}"
                        
        except Exception as e:
            self.logger.error(f"Error calling stock agent: {str(e)}", extra_data={
                'error': str(e),
                'task': task
            })
            return f"Failed to call stock agent: {str(e)}"

    async def process_message(self, message: str, session_id: str = None) -> Dict[str, Any]:
        """Override BaseAgent process_message to implement actual routing logic"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
                
            self.logger.info("Master Agent processing message with routing logic", extra_data={
                'message': message,
                'session_id': session_id,
                'step': 'master_agent_routing_start'
            })
            
            # First, ensure we have discovered agents
            await self._discover_agents("Initial discovery for task routing")
            
            # Route the task to appropriate agent
            routing_result = await self._route_task(message)
            
            self.logger.info("Task routing completed", extra_data={
                'routing_result': routing_result,
                'session_id': session_id,
                'step': 'master_agent_routing_complete'
            })
            
            return {
                'agent': self.agent_name,
                'response': routing_result,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Master Agent routing failed: {str(e)}", extra_data={
                'error': str(e),
                'session_id': session_id,
                'step': 'master_agent_routing_error'
            })
            return {
                'agent': self.agent_name,
                'error': str(e),
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }

    async def _get_agent_status(self, agent_name: str = "") -> str:
        """Get status of registered agents"""
        try:
            agents = self.registry.get_all_agents()
            
            if agent_name:
                # Get specific agent
                agent = next((a for a in agents if a.get('name') == agent_name), None)
                if agent:
                    return json.dumps(agent, indent=2)
                else:
                    return f"Agent '{agent_name}' not found"
            else:
                # Get all agents
                status_info = {
                    'total_agents': len(agents),
                    'agents': [
                        {
                            'name': agent.get('name'),
                            'status': agent.get('status'),
                            'capabilities': agent.get('capabilities', []),
                            'last_seen': agent.get('last_seen')
                        }
                        for agent in agents
                    ]
                }
                return json.dumps(status_info, indent=2)
                
        except Exception as e:
            return f"Failed to get agent status: {str(e)}"
    
    async def _coordinate_agents(self, coordination_request: str) -> str:
        """Coordinate multiple agents for complex tasks"""
        try:
            self.logger.info(f"Coordinating agents for: {coordination_request}", extra_data={
                'coordination_request': coordination_request
            })
            
            # Parse coordination request
            task_parts = await self._decompose_complex_task(coordination_request)
            
            # Assign tasks to appropriate agents
            assignments = []
            for part in task_parts:
                agent = await self._find_best_agent_for_task(part)
                if agent:
                    assignments.append({
                        'task': part,
                        'agent': agent.get('name'),
                        'agent_id': agent.get('id')
                    })
            
            # Execute coordinated tasks
            results = []
            for assignment in assignments:
                result = await self._send_task_to_agent(
                    {'name': assignment['agent'], 'id': assignment['agent_id']},
                    assignment['task']
                )
                results.append(f"{assignment['agent']}: {result}")
            
            return "Coordination completed:\n" + "\n".join(results)
            
        except Exception as e:
            return f"Coordination failed: {str(e)}"
    
    async def _decompose_complex_task(self, task: str) -> List[str]:
        """Decompose complex task into simpler subtasks"""
        prompt = f"""
        Decompose the following complex task into 2-4 simpler subtasks:
        
        Complex task: {task}
        
        Return only the subtasks, one per line, without numbering or additional text.
        """
        
        response = await self._get_llm_response(prompt, "task_decomposition")
        subtasks = [task.strip() for task in response.split('\n') if task.strip()]
        
        return subtasks
    
    async def _find_best_agent_for_task(self, task: str) -> Optional[Dict[str, Any]]:
        """Find the best agent for a specific task"""
        required_capabilities = await self._analyze_task_requirements(task)
        
        for capability in required_capabilities:
            agents = self.registry.get_agents_by_capability(capability)
            if agents:
                return agents[0]  # Return first available agent
        
        return None
    
    async def start_discovery_service(self):
        """Start the agent discovery service"""
        self.logger.info("Starting agent discovery service")
        
        # Start periodic discovery
        asyncio.create_task(self._periodic_discovery())
        
        # Start health monitoring
        asyncio.create_task(self._health_monitoring())
    
    async def _periodic_discovery(self):
        """Periodically discover new agents"""
        while True:
            try:
                await self._discover_agents("Periodic discovery scan")
                await asyncio.sleep(300)  # Discover every 5 minutes
            except Exception as e:
                self.logger.error(f"Periodic discovery failed: {str(e)}", extra_data={
                    'error': str(e)
                })
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def _health_monitoring(self):
        """Monitor health of registered agents"""
        while True:
            try:
                agents = self.registry.get_all_agents()
                for agent in agents:
                    # Simulate health check
                    await asyncio.sleep(0.1)
                    self.registry.update_agent_status(agent.get('id'), 'healthy')
                
                await asyncio.sleep(60)  # Check health every minute
            except Exception as e:
                self.logger.error(f"Health monitoring failed: {str(e)}", extra_data={
                    'error': str(e)
                })
                await asyncio.sleep(30)

# FastAPI server for master agent
app = FastAPI(title="Master Agent API", version="1.0.0")
master_agent = None

# Constants
MASTER_AGENT_NOT_INITIALIZED = "Master agent not initialized"

class TaskRequest(BaseModel):
    task: str
    session_id: Optional[str] = None

class AgentRegistrationRequest(BaseModel):
    name: str
    description: str
    capabilities: List[str]
    endpoint: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global master_agent
    master_agent = MasterAgent()
    await master_agent.start_discovery_service()

@app.post("/process_task")
async def process_task(request: TaskRequest):
    """Process a task through the master agent"""
    if not master_agent:
        master_agent.logger.error("Master agent not initialized for task processing")
        raise HTTPException(status_code=500, detail=MASTER_AGENT_NOT_INITIALIZED)
    
    try:
        master_agent.logger.info("Processing task request", extra_data={
            'task': request.task,
            'session_id': request.session_id,
            'request_type': 'task_processing'
        })
        
        # Log task analysis
        master_agent.logger.info("Analyzing task requirements", extra_data={
            'task_description': request.task,
            'step': 'task_analysis'
        })
        
        response = await master_agent.process_message(request.task, request.session_id)
        master_agent.logger.info("Response from master agent: " + str(response)[:200] + "...")
        
        master_agent.logger.info("Task processing completed successfully", extra_data={
            'task': request.task,
            'session_id': request.session_id,
            'response_length': len(str(response)),
            'response': response,
            'step': 'task_completion'
        })
        
        return response
    except Exception as e:
        master_agent.logger.error("Task processing failed", extra_data={
            'task': request.task,
            'session_id': request.session_id,
            'error': str(e),
            'error_type': type(e).__name__,
            'step': 'task_error'
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register_agent")
async def register_agent(request: AgentRegistrationRequest):
    """Register a new agent"""
    if not master_agent:
        raise HTTPException(status_code=500, detail=MASTER_AGENT_NOT_INITIALIZED)
    
    try:
        agent_info = {
            'id': str(uuid.uuid4()),
            'name': request.name,
            'description': request.description,
            'capabilities': request.capabilities,
            'endpoint': request.endpoint,
            'status': 'active'
        }
        
        agent_id = master_agent.registry.register_agent(agent_info)
        return {"agent_id": agent_id, "status": "registered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def get_agents():
    """Get all registered agents"""
    if not master_agent:
        raise HTTPException(status_code=500, detail=MASTER_AGENT_NOT_INITIALIZED)
    
    return {"agents": master_agent.registry.get_all_agents()}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not master_agent:
        return {"status": "unhealthy", "error": MASTER_AGENT_NOT_INITIALIZED}
    
    try:
        health = await master_agent.health_check()
        return health
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv('MASTER_AGENT_PORT', 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
