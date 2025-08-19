import asyncio
import json
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
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
        
        # Scope and intent management
        self.scope_boundaries = {
            'allowed_domains': [],
            'forbidden_actions': [],
            'max_iterations': 10,
            'timeout_seconds': 300,
            'rate_limits': {
                'requests_per_minute': 60,
                'requests_per_hour': 1000
            }
        }
        
        # Safety and validation
        self.safety_checks = {
            'input_validation': True,
            'output_filtering': True,
            'intent_verification': True,
            'scope_enforcement': True
        }
        
        # Rate limiting
        self.request_history = []
        
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
        """Process a message and return response with scope validation"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Record request for rate limiting
            self.request_history.append(datetime.now())
            
            self.logger.info("Processing message", extra_data={
                'message_length': len(message),
                'session_id': session_id,
                'agent_name': self.agent_name,
                'step': 'message_received'
            })
            
            # Input validation
            if self.safety_checks['input_validation']:
                validation_result = self.validate_input(message)
                if not validation_result['is_valid']:
                    self.logger.warning(f"Input validation failed for {self.agent_name}", extra_data={
                        'errors': validation_result['errors'],
                        'session_id': session_id
                    })
                    return {
                        "agent": self.agent_name,
                        "error": f"Input validation failed: {'; '.join(validation_result['errors'])}",
                        "warnings": validation_result['warnings'],
                        "session_id": session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                
                if validation_result['warnings']:
                    self.logger.warning(f"Input validation warnings for {self.agent_name}", extra_data={
                        'warnings': validation_result['warnings'],
                        'session_id': session_id
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
            
            # Intent verification
            intent_result = {'aligned': True, 'confidence': 1.0, 'concerns': []}
            if self.safety_checks['intent_verification']:
                intent_result = self.verify_intent(message, response)
                if not intent_result['aligned']:
                    self.logger.warning(f"Intent verification failed for {self.agent_name}", extra_data={
                        'concerns': intent_result['concerns'],
                        'confidence': intent_result['confidence'],
                        'session_id': session_id
                    })
                    # Modify response to include warning
                    response = f"⚠️ SCOPE WARNING: {intent_result['concerns'][0]}\n\n{response}"
            
            # Output filtering
            if self.safety_checks['output_filtering']:
                response = self.filter_output(response)
            
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
                'timestamp': datetime.now().isoformat(),
                'validation': {
                    'input_valid': validation_result.get('is_valid', True),
                    'intent_aligned': intent_result.get('aligned', True),
                    'confidence': intent_result.get('confidence', 1.0)
                }
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

    def set_scope_boundaries(self, boundaries: Dict[str, Any]):
        """Set scope boundaries for the agent"""
        self.scope_boundaries.update(boundaries)
        self.logger.info(f"Scope boundaries updated for {self.agent_name}", extra_data={
            'boundaries': boundaries
        })
    
    def add_forbidden_action(self, action: str):
        """Add a forbidden action to the scope boundaries"""
        self.scope_boundaries['forbidden_actions'].append(action)
        self.logger.info(f"Forbidden action added: {action}", extra_data={
            'action': action
        })
    
    def add_allowed_domain(self, domain: str):
        """Add an allowed domain to the scope boundaries"""
        self.scope_boundaries['allowed_domains'].append(domain)
        self.logger.info(f"Allowed domain added: {domain}", extra_data={
            'domain': domain
        })
    
    def validate_input(self, message: str) -> Dict[str, Any]:
        """Validate input against scope boundaries"""
        validation_result = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }
        
        # Check for forbidden actions
        for forbidden_action in self.scope_boundaries['forbidden_actions']:
            if forbidden_action.lower() in message.lower():
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Forbidden action detected: {forbidden_action}")
        
        # Check domain relevance
        if self.scope_boundaries['allowed_domains']:
            domain_match = False
            for domain in self.scope_boundaries['allowed_domains']:
                if domain.lower() in message.lower():
                    domain_match = True
                    break
            
            if not domain_match:
                validation_result['warnings'].append("Input may be outside agent's primary domain")
        
        # Rate limiting check
        if not self._check_rate_limits():
            validation_result['is_valid'] = False
            validation_result['errors'].append("Rate limit exceeded")
        
        return validation_result
    
    def _check_rate_limits(self) -> bool:
        """Check if rate limits are exceeded"""
        now = datetime.now()
        minute_ago = now.replace(second=0, microsecond=0) - timedelta(minutes=1)
        hour_ago = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
        
        # Clean old requests
        self.request_history = [req for req in self.request_history if req > minute_ago]
        
        # Check minute limit
        minute_requests = len([req for req in self.request_history if req > minute_ago])
        if minute_requests >= self.scope_boundaries['rate_limits']['requests_per_minute']:
            return False
        
        # Check hour limit
        hour_requests = len([req for req in self.request_history if req > hour_ago])
        if hour_requests >= self.scope_boundaries['rate_limits']['requests_per_hour']:
            return False
        
        return True
    
    def verify_intent(self, message: str, response: str) -> Dict[str, Any]:
        """Verify that the response aligns with the agent's intent"""
        intent_result = {
            'aligned': True,
            'confidence': 0.0,
            'concerns': []
        }
        
        # Check if response stays within agent's description
        agent_keywords = self.agent_description.lower().split()
        response_lower = response.lower()
        
        # Calculate alignment score
        keyword_matches = sum(1 for keyword in agent_keywords if keyword in response_lower)
        alignment_score = keyword_matches / len(agent_keywords) if agent_keywords else 0
        
        intent_result['confidence'] = alignment_score
        
        if alignment_score < 0.3:
            intent_result['aligned'] = False
            intent_result['concerns'].append("Response appears to deviate from agent's primary purpose")
        
        # Check for scope violations
        if self.scope_boundaries['forbidden_actions']:
            for forbidden_action in self.scope_boundaries['forbidden_actions']:
                if forbidden_action.lower() in response_lower:
                    intent_result['aligned'] = False
                    intent_result['concerns'].append(f"Response contains forbidden action: {forbidden_action}")
        
        return intent_result
    
    def filter_output(self, response: str) -> str:
        """Filter and sanitize output"""
        if not self.safety_checks['output_filtering']:
            return response
        
        # Remove potentially sensitive information
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        ]
        
        import re
        for pattern in sensitive_patterns:
            response = re.sub(pattern, '[REDACTED]', response)
        
        return response
