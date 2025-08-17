# AgentEduverse - Setup Guide

This guide will help you set up and run the comprehensive multi-agent AI system for stock analysis and investment recommendations.

## 🏗️ System Architecture

The system consists of four main components:

1. **MCP Server** (Port 8003) - Finnhub API integration
2. **Master Agent** (Port 8001) - Agent coordination and discovery
3. **Stock Agent** (Port 8002) - Stock analysis and recommendations
4. **Streamlit Frontend** (Port 8501) - User interface

## 📋 Prerequisites

### Required Software
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Required API Keys
- **Finnhub API Key** - Get from [https://finnhub.io/](https://finnhub.io/)
- **AWS Access Key ID** - For AWS Bedrock access
- **AWS Secret Access Key** - For AWS Bedrock access
- **LangSmith API Key** - Get from [https://smith.langchain.com/](https://smith.langchain.com/)

## 🚀 Installation Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd AgentEduverse
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```bash
# Copy example configuration
cp config.env.example .env
```

Edit the `.env` file with your API keys:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1

# Finnhub API
FINNHUB_API_KEY=your_finnhub_api_key

# LangSmith Configuration
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agent-eduverse

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/multi_agent_system.log

# Agent Configuration
MASTER_AGENT_PORT=8001
STOCK_AGENT_PORT=8002
MCP_SERVER_PORT=8003

# Streamlit Configuration
STREAMLIT_SERVER_PORT=8501
```

### 5. Create Logs Directory
```bash
mkdir logs
```

## 🏃‍♂️ Running the System

### Option 1: Start All Components (Recommended)
```bash
python start_system.py
```

This will start all components in the correct order and provide health checks.

### Option 2: Start Components Individually

#### Start MCP Server
```bash
python mcp_server/finnhub_server.py
```

#### Start Master Agent
```bash
python agents/master_agent.py
```

#### Start Stock Agent
```bash
python agents/stock_agent.py
```

#### Start Streamlit Frontend
```bash
streamlit run frontend/app.py
```

### Option 3: Start Specific Components
```bash
# Start only MCP server
python start_system.py --mcp-only

# Start only agents (no frontend)
python start_system.py --agents-only

# Start only frontend
python start_system.py --frontend-only

# Start with monitoring
python start_system.py --monitor
```

## 🌐 Accessing the System

Once all components are running, you can access:

- **Frontend Interface**: http://localhost:8501
- **Master Agent API**: http://localhost:8001
- **Stock Agent API**: http://localhost:8002
- **MCP Server API**: http://localhost:8003

## 🧪 Testing the System

### Run All Tests
```bash
python test_system.py
```

### Test Specific Components
```bash
# Test MCP server only
python test_system.py --component mcp

# Test master agent only
python test_system.py --component master

# Test stock agent only
python test_system.py --component stock

# Test frontend only
python test_system.py --component frontend

# Test integration only
python test_system.py --component integration
```

### Generate Test Report
```bash
python test_system.py --output test_report.txt
```

## 📊 Using the System

### Via Streamlit Frontend

1. Open http://localhost:8501 in your browser
2. Use the chat interface to ask questions about stocks
3. Try example queries like:
   - "Analyze Apple stock and give me a buy/sell recommendation"
   - "What's the technical analysis for Tesla?"
   - "Compare Microsoft and Google stocks"

### Via API Endpoints

#### Master Agent API
```bash
# Process a task
curl -X POST http://localhost:8001/process_task \
  -H "Content-Type: application/json" \
  -d '{"task": "Analyze AAPL stock", "session_id": "test"}'

# Get registered agents
curl http://localhost:8001/agents

# Health check
curl http://localhost:8001/health
```

#### Stock Agent API
```bash
# Analyze stock
curl -X POST http://localhost:8002/analyze_stock \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "analysis_type": "comprehensive"}'

# Health check
curl http://localhost:8002/health
```

#### MCP Server API
```bash
# Get available tools
curl http://localhost:8003/tools

# Execute tool
curl -X POST http://localhost:8003/execute \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "get_stock_quote", "parameters": {"symbol": "AAPL"}}'

# Health check
curl http://localhost:8003/health
```

## 🔧 Configuration Options

### Logging Configuration
- `LOG_LEVEL`: Set to DEBUG, INFO, WARNING, ERROR, or CRITICAL
- `LOG_FILE`: Path to the log file

### Port Configuration
- `MASTER_AGENT_PORT`: Master agent API port (default: 8001)
- `STOCK_AGENT_PORT`: Stock agent API port (default: 8002)
- `MCP_SERVER_PORT`: MCP server API port (default: 8003)
- `STREAMLIT_SERVER_PORT`: Frontend port (default: 8501)

### AWS Configuration
- `AWS_REGION`: AWS region for Bedrock (default: us-east-1)

## 🐛 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Check what's using the port
lsof -i :8001  # For master agent port
lsof -i :8002  # For stock agent port
lsof -i :8003  # For MCP server port
lsof -i :8501  # For frontend port

# Kill the process
kill -9 <PID>
```

#### 2. Missing API Keys
Ensure all required API keys are set in your `.env` file:
- `FINNHUB_API_KEY`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `LANGCHAIN_API_KEY`

#### 3. AWS Bedrock Access
Make sure your AWS account has access to Amazon Bedrock and the required models.

#### 4. Network Connectivity
Ensure all components can communicate with each other on localhost.

### Logs
Check the logs directory for detailed error information:
```bash
tail -f logs/multi_agent_system.log
```

### Health Checks
Test individual component health:
```bash
curl http://localhost:8001/health  # Master agent
curl http://localhost:8002/health  # Stock agent
curl http://localhost:8003/health  # MCP server
```

## 🔄 Adding New Agents

To add a new agent to the system:

1. Create a new agent class inheriting from `BaseAgent`
2. Implement the required methods
3. Register the agent with the master agent
4. Update the discovery logic in the master agent

Example:
```python
from agents.base_agent import BaseAgent

class NewsAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            agent_name="NewsAgent",
            agent_description="Agent for news analysis and sentiment processing"
        )
        self.add_capability("news_analysis")
        self.add_capability("sentiment_analysis")
```

## 📈 Scaling the System

The system is designed to be scalable:

1. **Horizontal Scaling**: Run multiple instances of each agent
2. **Load Balancing**: Use a load balancer for agent distribution
3. **Database Integration**: Add persistent storage for agent state
4. **Message Queues**: Implement async message processing
5. **Containerization**: Use Docker for easy deployment

## 🔒 Security Considerations

1. **API Key Management**: Use environment variables or secure key management
2. **Network Security**: Implement proper authentication and authorization
3. **Rate Limiting**: Add rate limiting to prevent API abuse
4. **Input Validation**: Validate all user inputs
5. **Logging**: Monitor system logs for security events

## 📝 API Documentation

For detailed API documentation, visit:
- Master Agent: http://localhost:8001/docs
- Stock Agent: http://localhost:8002/docs
- MCP Server: http://localhost:8003/docs

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the troubleshooting section
2. Review the logs
3. Run the test suite
4. Create an issue in the repository

---

**Happy Trading! 📈🤖**
