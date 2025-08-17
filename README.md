# AgentEduverse

A comprehensive multi-agent AI system with a master agent that discovers and coordinates specialized agents, each with unique capabilities and data sources. Built with LangChain, LangGraph, and AWS Bedrock for advanced AI orchestration.

## 🏗️ Architecture Overview

The system follows a distributed microservices architecture with the following key components:

### Core Components

1. **Master Agent** (`agents/master_agent.py`)
   - **Port**: 8001 (configurable via `MASTER_AGENT_PORT`)
   - **Role**: Central coordinator and task router
   - **Capabilities**: Agent discovery, task routing, coordination, load balancing, health monitoring
   - **Features**: 
     - Simulated A2A (Agent-to-Agent) discovery framework
     - Intelligent task routing based on agent capabilities
     - Real-time agent health monitoring
     - Periodic agent discovery (every 5 minutes)
     - RESTful API endpoints for task processing

2. **Stock Agent** (`agents/stock_agent.py`)
   - **Port**: 8002 (configurable via `STOCK_AGENT_PORT`)
   - **Role**: Specialized financial analysis and investment recommendations
   - **Capabilities**: Stock analysis, financial data processing, technical analysis, news analysis, investment recommendations
   - **Features**:
     - Comprehensive stock analysis engine
     - Real-time price data and financial metrics
     - Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
     - News sentiment analysis
     - Buy/Sell/Hold recommendations with confidence scores
     - Financial statement analysis

3. **MCP Server** (`mcp_server/finnhub_server.py`)
   - **Port**: 8003 (configurable via `MCP_SERVER_PORT`)
   - **Role**: Model Context Protocol server for Finnhub API integration
   - **Capabilities**: Centralized data source management
   - **Available Tools**:
     - `get_stock_quote`: Real-time stock quotes
     - `get_company_profile`: Company profiles and metrics
     - `get_financial_statements`: Income, balance, and cash flow statements
     - `get_news`: Company news and sentiment
     - `get_analyst_recommendations`: Analyst ratings and trends
     - `get_earnings_calendar`: Earnings announcements
     - `get_insider_transactions`: Insider trading data
     - `get_technical_indicators`: Technical analysis calculations

4. **Streamlit Frontend** (`frontend/app.py`)
   - **Port**: 8501 (configurable via `STREAMLIT_SERVER_PORT`)
   - **Role**: User interface and chatbot
   - **Features**:
     - Modern chat interface
     - Real-time system health monitoring
     - Session management
     - Responsive design
     - Error handling and user feedback

5. **Base Agent** (`agents/base_agent.py`)
   - **Role**: Foundation class for all agents
   - **Features**:
     - AWS Bedrock LLM integration (Claude 3 Sonnet)
     - LangChain tool management
     - Comprehensive logging
     - Health monitoring
     - State management

6. **Logging System** (`utils/logger.py`)
   - **Role**: Centralized logging with multiple formatters
   - **Features**:
     - Console logging with colors
     - Structured JSON file logging
     - Session-based logging
     - Rotating file handlers
     - Multi-level logging (DEBUG, INFO, WARNING, ERROR)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- AWS Account with Bedrock access
- Finnhub API key
- LangSmith API key (optional for monitoring)

### Installation

1. **Clone and setup**:
```bash
git clone <repository-url>
cd AgentEduverse
pip install -r requirements.txt
```

2. **Environment Configuration**:
```bash
cp .env.example .env
# Edit .env with your credentials:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_REGION
# - FINNHUB_API_KEY
# - LANGCHAIN_API_KEY (optional)
# - LANGCHAIN_TRACING_V2 (optional)
# - LANGCHAIN_PROJECT (optional)
```

3. **Start the System**:
```bash
# Option 1: Start all components at once
python start_system.py

# Option 2: Start components individually
python mcp_server/finnhub_server.py &
python agents/master_agent.py &
python agents/stock_agent.py &
streamlit run frontend/app.py
```

4. **Access the Application**:
- Frontend: http://localhost:8501
- Master Agent API: http://localhost:8001
- Stock Agent API: http://localhost:8002
- MCP Server API: http://localhost:8003

## 🔧 System Features

### Multi-Agent Coordination

- **Agent Discovery**: Automatic discovery and registration of available agents
- **Task Routing**: Intelligent routing based on agent capabilities
- **Load Balancing**: Distributed task processing
- **Health Monitoring**: Real-time agent status tracking
- **Fault Tolerance**: Graceful error handling and recovery

### Stock Analysis Capabilities

- **Real-time Data**: Live stock quotes and market data
- **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands
- **Fundamental Analysis**: Financial statements, ratios, metrics
- **News Analysis**: Sentiment analysis of market news
- **Recommendation Engine**: AI-powered buy/sell/hold recommendations
- **Risk Assessment**: Confidence scoring and risk evaluation

### Data Integration

- **Finnhub APIs**: Comprehensive financial data
- **AWS Bedrock**: Advanced LLM capabilities
- **LangChain**: Agent orchestration and tool management
- **LangGraph**: Workflow management and state handling
- **LangSmith**: Monitoring and tracing (optional)

### Logging and Monitoring

- **Centralized Logging**: Single log file for all components
- **Console Output**: Real-time logging with colors
- **Structured Logs**: JSON format for analysis
- **Session Tracking**: Request tracing across components
- **Error Handling**: Comprehensive error logging and recovery

## 📊 API Endpoints

### Master Agent (Port 8001)

- `POST /process_task`: Process tasks through agent routing
- `POST /register_agent`: Register new agents
- `GET /agents`: List all registered agents
- `GET /health`: Health check

### Stock Agent (Port 8002)

- `POST /analyze_stock`: Comprehensive stock analysis
- `GET /health`: Health check

### MCP Server (Port 8003)

- `POST /execute`: Execute MCP tools
- `GET /tools`: List available tools
- `GET /health`: Health check

## 🛠️ Development and Testing

### Testing the System

```bash
# Test individual components
python test_system.py

# Test specific functionality
python -c "from agents.stock_agent import StockAgent; print('Stock Agent imported successfully')"
```

### Adding New Agents

1. Create a new agent class inheriting from `BaseAgent`
2. Implement required methods and capabilities
3. Register the agent with the Master Agent
4. Update the discovery logic if needed

### Adding New Data Sources

1. Extend the MCP Server with new tools
2. Update the tool definitions and execution logic
3. Integrate with existing agents as needed

## 🔍 Troubleshooting

### Common Issues

1. **Environment Variables**: Ensure all required API keys are set in `.env`
2. **Port Conflicts**: Check that ports 8001-8003 and 8501 are available
3. **API Limits**: Monitor Finnhub API usage and rate limits
4. **AWS Credentials**: Verify AWS Bedrock access and credentials

### Log Analysis

- Check console output for real-time logs
- Review log files in the project directory
- Use session IDs to trace request flows
- Monitor agent health status

## 📈 Performance and Scalability

### Current Capabilities

- **Concurrent Processing**: Multiple agents can process tasks simultaneously
- **Modular Design**: Easy to add new agents and data sources
- **Error Recovery**: Graceful handling of API failures and timeouts
- **Mock Data Fallbacks**: System continues working even with API issues

### Scalability Features

- **Stateless Design**: Agents can be scaled horizontally
- **Message Queuing**: Asynchronous task processing
- **Load Distribution**: Intelligent task routing
- **Health Monitoring**: Automatic detection of failed components

## 🤝 Contributing

### 🚀 Join Our Community of AI Innovators!

We're excited to welcome passionate developers, researchers, and AI enthusiasts to help shape the future of multi-agent AI systems! This project represents a cutting-edge approach to distributed AI coordination, and your contributions can make a real impact.

### 🌟 Why Contribute?

- **Pioneer AI Technology**: Be part of developing next-generation multi-agent systems that could revolutionize how AI agents collaborate
- **Learn Advanced Concepts**: Work with state-of-the-art technologies like LangChain, LangGraph, AWS Bedrock, and Model Context Protocol
- **Build Real-World Skills**: Gain hands-on experience with distributed systems, API integration, and AI orchestration
- **Open Source Impact**: Contribute to an educational platform that helps others learn about AI agent systems
- **Community Recognition**: Get your name in our contributors list and build your portfolio

### 🎯 Areas Where We Need Your Help

**For Developers:**
- 🛠️ **New Agent Types**: Create specialized agents (News Agent, Technical Analysis Agent, Sentiment Analysis Agent)
- 🔧 **System Improvements**: Enhance error handling, performance optimization, and scalability
- 🎨 **Frontend Enhancements**: Improve the Streamlit interface with better UX/UI
- 📊 **Data Visualization**: Add charts, graphs, and interactive dashboards
- 🔒 **Security Features**: Implement authentication, authorization, and data protection

**For Researchers:**
- 🧠 **AI Algorithm Improvements**: Enhance recommendation engines and analysis algorithms
- 📈 **Performance Metrics**: Add comprehensive benchmarking and evaluation tools
- 🔬 **Experimental Features**: Test new AI coordination patterns and communication protocols
- 📚 **Documentation**: Improve technical documentation and research papers

**For Everyone:**
- 🐛 **Bug Fixes**: Help identify and resolve issues
- 📖 **Documentation**: Improve guides, tutorials, and code comments
- 💡 **Feature Ideas**: Suggest new capabilities and improvements
- 🌍 **Community Building**: Help others learn and contribute

### 🛠️ How to Get Started

1. **Fork the repository** and clone it to your local machine
2. **Set up the development environment** following our setup guide
3. **Pick an issue** from our [Issues](https://github.com/your-repo/issues) page or create your own
4. **Create a feature branch** with a descriptive name
5. **Implement your changes** with proper testing and documentation
6. **Submit a pull request** with a clear description of your improvements

### 🎉 Recognition

- **Contributors Hall of Fame**: Your name will be featured in our contributors list
- **Code Review**: Get feedback from experienced developers
- **Mentorship**: Connect with other contributors and learn from each other
- **Showcase**: Highlight your contributions in your portfolio and resume

### 💬 Get in Touch

- **Discussions**: Join our [GitHub Discussions](https://github.com/your-repo/discussions) for questions and ideas
- **Issues**: Report bugs or suggest features via [GitHub Issues](https://github.com/your-repo/issues)
- **Community**: Connect with other contributors and share your experiences

### 🌱 Beginner-Friendly

Don't worry if you're new to AI or multi-agent systems! We welcome contributors of all skill levels:
- **First-time contributors**: Start with documentation or simple bug fixes
- **Learning-focused**: Use this project to learn about AI agent systems
- **Experienced developers**: Help mentor others and tackle complex features

**Every contribution, no matter how small, makes a difference!** 🎯

---

*Ready to make your mark in the AI revolution? Let's build the future of multi-agent systems together!* 🚀

## 📄 License

This project is licensed under the MIT License:

```
MIT License

Copyright (c) 2025 AgentEduverse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**Important Disclaimers:**

- **Educational Purpose Only**: This software is developed and distributed solely for educational and research purposes.
- **No Financial Advice**: The stock analysis and recommendations provided by this system are for educational demonstration only and should not be considered as financial advice.
- **No Warranty**: The software is provided "as is" without any warranties of any kind.
- **Open Source**: This is open-source software that can be freely used, modified, and distributed under the MIT License terms.
- **Professional Consultation**: Always consult with qualified financial advisors before making any investment decisions.
- **API Limitations**: The system relies on external APIs (Finnhub, AWS Bedrock) which may have usage limits and terms of service.
- **Experimental Nature**: This is an experimental AI system and should not be used in production environments without proper testing and validation.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs for error details
3. Open an issue with detailed information
4. Include relevant log snippets and error messages

---

**Note**: This system is designed for educational and research purposes. Always verify financial recommendations with professional advisors before making investment decisions.
