import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime
import boto3
from langchain.llms import Bedrock
from langchain.chat_models import BedrockChat
from langchain.schema import HumanMessage, SystemMessage
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langsmith import Client
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger

class BaseAgent:
    """Base class for all agents in the AgentEduverse system"""
    
    def __init__(self, agent_name: str, agent_description: str):
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.logger = logger.get_logger(agent_name)
        
        # Initialize AWS Bedrock client
        self.bedrock_client = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        
        # Initialize LangSmith client
        self.langsmith_client = Client()
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Agent state
        self.state = {}
        self.tools = []
        self.capabilities = []
        
        self.logger.info(f"Base agent {agent_name} initialized", extra_data={
            'agent_name': agent_name,
            'description': agent_description
        })
    
    def _initialize_llm(self) -> BedrockChat:
        """Initialize AWS Bedrock LLM"""
        try:
            llm = BedrockChat(
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                client=self.bedrock_client,
                model_kwargs={
                    "temperature": 0.1,
                    "max_tokens": 4000
                }
            )
            self.logger.info("LLM initialized successfully", extra_data={
                'model': "anthropic.claude-3-sonnet-20240229-v1:0"
            })
            return llm
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}", extra_data={
                'error': str(e)
            })
            raise
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent"""
        self.tools.append(tool)
        self.logger.info(f"Tool added: {tool.name}", extra_data={
            'tool_name': tool.name,
            'tool_description': tool.description
        })
    
    def add_capability(self, capability: str):
        """Add a capability to the agent"""
        self.capabilities.append(capability)
        self.logger.info(f"Capability added: {capability}", extra_data={
            'capability': capability
        })
    
    async def process_message(self, message: str, session_id: str = None) -> Dict[str, Any]:
        """Process a message and return response"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
                
            self.logger.info("Processing message", extra_data={
                'message_length': len(message),
                'session_id': session_id,
                'agent_name': self.agent_name,
                'step': 'message_received'
            })
            
            # Log message content for debugging
            self.logger.debug("Message content", extra_data={
                'message': message[:200] + "..." if len(message) > 200 else message,
                'session_id': session_id,
                'step': 'message_content'
            })
            
            # Create agent prompt
            self.logger.info("Creating agent prompt", extra_data={
                'session_id': session_id,
                'capabilities_count': len(self.capabilities),
                'tools_count': len(self.tools),
                'step': 'prompt_creation'
            })
            
            prompt = self._create_prompt(message)
            self.logger.info(f"Created Prompt: {prompt}")
            # Get LLM response
            self.logger.info("Requesting LLM response", extra_data={
                'session_id': session_id,
                'prompt_length': len(prompt),
                'step': 'llm_request'
            })
            self.logger.info("sending prompt to llm to get response and then decide which agent to use")
            response = await self._get_llm_response(prompt, session_id)
            self.logger.info(f"Response from LLM: {response}")
            
            self.logger.info("LLM response received", extra_data={
                'session_id': session_id,
                'response_length': len(response),
                'step': 'llm_response'
            })
            
            # Update state
            self.logger.info("Updating state")
            self.state['last_message'] = message
            self.state['last_response'] = response
            self.state['last_processed'] = datetime.now().isoformat()
            self.logger.info(f"State: {self.state}")
            self.logger.info("Message processing completed", extra_data={
                'session_id': session_id,
                'response_length': len(response),
                'step': 'processing_complete'
            })

            self.logger.info("Returning response")
            return {
                'agent': self.agent_name,
                'response': response,
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing message: {str(e)}", extra_data={
                'error': str(e),
                'error_type': type(e).__name__,
                'session_id': session_id,
                'step': 'processing_error'
            })
            return {
                'agent': self.agent_name,
                'error': str(e),
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_prompt(self, message: str) -> str:
        """Create a prompt for the agent"""
        return f"""
        You are {self.agent_name}, a specialized AI agent with the following description:
        {self.agent_description}
        
        Your capabilities include:
        {', '.join(self.capabilities)}
        
        Available tools:
        {[tool.name for tool in self.tools]}
        
        User message: {message}
        
        Please provide a helpful and accurate response based on your capabilities and available tools.
        """
    
    async def _get_llm_response(self, prompt: str, session_id: str = None) -> str:
        """Get response from LLM"""
        try:
            messages = [
                SystemMessage(content="You are a helpful AI assistant with specialized capabilities."),
                HumanMessage(content=prompt)
            ]
            
            response = await self.llm.agenerate([messages])
            return response.generations[0][0].text
            
        except Exception as e:
            self.logger.error(f"LLM response error: {str(e)}", extra_data={
                'error': str(e),
                'session_id': session_id
            })
            return f"I apologize, but I encountered an error: {str(e)}"
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information for discovery"""
        return {
            'name': self.agent_name,
            'description': self.agent_description,
            'capabilities': self.capabilities,
            'tools': [tool.name for tool in self.tools],
            'status': 'active',
            'timestamp': datetime.now().isoformat()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for the agent"""
        try:
            # Test LLM connection
            test_response = await self._get_llm_response("Hello", "health_check")
            
            return {
                'agent': self.agent_name,
                'status': 'healthy',
                'llm_working': bool(test_response),
                'tools_count': len(self.tools),
                'capabilities_count': len(self.capabilities),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'agent': self.agent_name,
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info(f"Cleaning up agent {self.agent_name}")
        # Add any cleanup logic here
