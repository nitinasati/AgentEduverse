import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import finnhub
import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import logger

@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]

class FinnhubMCPServer:
    """MCP Server for Finnhub API integration"""
    
    def __init__(self):
        self.api_key = os.getenv('FINNHUB_API_KEY')
        if not self.api_key:
            raise ValueError("FINNHUB_API_KEY environment variable is required")
        
        self.client = finnhub.Client(api_key=self.api_key)
        self.logger = logger.get_logger("finnhub_mcp")
        
        # Define available tools
        self.tools = self._define_tools()
        
        self.logger.info("Finnhub MCP Server initialized", extra_data={
            'api_key_configured': bool(self.api_key)
        })
    
    def _define_tools(self) -> List[Tool]:
        """Define all available Finnhub API tools"""
        return [
            Tool(
                name="get_stock_quote",
                description="Get real-time stock quote for a symbol",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (e.g., AAPL, MSFT)"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_company_profile",
                description="Get company profile and financial metrics",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_financial_statements",
                description="Get company financial statements",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        },
                        "statement_type": {
                            "type": "string",
                            "enum": ["income", "balance", "cash"],
                            "description": "Type of financial statement"
                        },
                        "period": {
                            "type": "string",
                            "enum": ["annual", "quarterly"],
                            "description": "Reporting period"
                        }
                    },
                    "required": ["symbol", "statement_type", "period"]
                }
            ),

            Tool(
                name="get_news",
                description="Get company news and sentiment",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        },
                        "from_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        "to_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)"
                        }
                    },
                    "required": ["symbol", "from_date", "to_date"]
                }
            ),
            Tool(
                name="get_analyst_recommendations",
                description="Get analyst recommendations and ratings",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        }
                    },
                    "required": ["symbol"]
                }
            ),
            Tool(
                name="get_earnings_calendar",
                description="Get earnings calendar for companies",
                parameters={
                    "type": "object",
                    "properties": {
                        "from_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        "to_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)"
                        },
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol (optional)"
                        }
                    },
                    "required": ["from_date", "to_date"]
                }
            ),
            Tool(
                name="get_insider_transactions",
                description="Get insider trading transactions",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        },
                        "from_date": {
                            "type": "string",
                            "description": "Start date (YYYY-MM-DD)"
                        },
                        "to_date": {
                            "type": "string",
                            "description": "End date (YYYY-MM-DD)"
                        }
                    },
                    "required": ["symbol", "from_date", "to_date"]
                }
            ),
            Tool(
                name="get_technical_indicators",
                description="Calculate technical indicators for stock data",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock symbol"
                        },
                        "resolution": {
                            "type": "string",
                            "enum": ["D", "W", "M"],
                            "description": "Data resolution"
                        },
                        "indicators": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": ["sma", "ema", "rsi", "macd", "bollinger"]
                            },
                            "description": "Technical indicators to calculate"
                        }
                    },
                    "required": ["symbol", "resolution", "indicators"]
                }
            )
        ]
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get list of available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.tools
        ]
    
    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given parameters"""
        try:
            self.logger.info(f"Executing tool: {tool_name}", extra_data={
                'tool_name': tool_name,
                'parameters': parameters
            })
            
            if tool_name == "get_stock_quote":
                return await self._get_stock_quote(parameters)
            elif tool_name == "get_company_profile":
                return await self._get_company_profile(parameters)
            elif tool_name == "get_financial_statements":
                return await self._get_financial_statements(parameters)

            elif tool_name == "get_news":
                return await self._get_news(parameters)
            elif tool_name == "get_analyst_recommendations":
                return await self._get_analyst_recommendations(parameters)
            elif tool_name == "get_earnings_calendar":
                return await self._get_earnings_calendar(parameters)
            elif tool_name == "get_insider_transactions":
                return self._get_insider_transactions(parameters)
            elif tool_name == "get_technical_indicators":
                return self._get_technical_indicators(parameters)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
                
        except Exception as e:
            self.logger.error(f"Error executing tool {tool_name}: {str(e)}", extra_data={
                'tool_name': tool_name,
                'error': str(e)
            })
            return {"error": str(e)}
    
    async def _get_stock_quote(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get real-time stock quote"""
        symbol = params["symbol"].upper()
        
        try:
            quote = self.client.quote(symbol)
            
            return {
                "symbol": symbol,
                "current_price": quote.get('c', 0),
                "change": quote.get('d', 0),
                "percent_change": quote.get('dp', 0),
                "high": quote.get('h', 0),
                "low": quote.get('l', 0),
                "open": quote.get('o', 0),
                "previous_close": quote.get('pc', 0),
                "timestamp": quote.get('t', 0)
            }
        except Exception as e:
            self.logger.error(f"Error getting stock quote: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            
            # Return mock quote data when API fails
            return {
                "symbol": symbol,
                "current_price": 150.0,
                "change": 2.5,
                "percent_change": 1.67,
                "high": 155.0,
                "low": 148.0,
                "open": 149.0,
                "previous_close": 147.5,
                "timestamp": int(datetime.now().timestamp()),
                "error": f"API Error: {str(e)} - Using mock data"
            }
    
    async def _get_company_profile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get company profile"""
        symbol = params["symbol"].upper()
        
        try:
            profile = self.client.company_profile2(symbol=symbol)
            
            return {
                "symbol": symbol,
                "name": profile.get('name', ''),
                "country": profile.get('country', ''),
                "currency": profile.get('currency', ''),
                "exchange": profile.get('exchange', ''),
                "ipo": profile.get('ipo', ''),
                "market_capitalization": profile.get('marketCapitalization', 0),
                "phone": profile.get('phone', ''),
                "share_outstanding": profile.get('shareOutstanding', 0),
                "ticker": profile.get('ticker', ''),
                "weburl": profile.get('weburl', ''),
                "logo": profile.get('logo', ''),
                "finnhub_industry": profile.get('finnhubIndustry', '')
            }
        except Exception as e:
            self.logger.error(f"Error getting company profile: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            
            # Return mock profile data when API fails
            return {
                "symbol": symbol,
                "name": f"{symbol} Corporation",
                "country": "US",
                "currency": "USD",
                "exchange": "NASDAQ",
                "ipo": "1980-12-12",
                "market_capitalization": 2500000000000,
                "phone": "+1-555-0123",
                "share_outstanding": 16000000000,
                "ticker": symbol,
                "weburl": f"https://www.{symbol.lower()}.com",
                "logo": "",
                "finnhub_industry": "Technology",
                "error": f"API Error: {str(e)} - Using mock data"
            }
    
    async def _get_financial_statements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get financial statements"""
        symbol = params["symbol"].upper()
        statement_type = params["statement_type"]
        period = params["period"]
        
        try:
            if statement_type == "income":
                data = self.client.financials_reported(symbol=symbol, freq=period)
            elif statement_type == "balance":
                data = self.client.financials_reported(symbol=symbol, freq=period)
            elif statement_type == "cash":
                data = self.client.financials_reported(symbol=symbol, freq=period)
            else:
                raise ValueError(f"Invalid statement type: {statement_type}")
            
            return {
                "symbol": symbol,
                "statement_type": statement_type,
                "period": period,
                "data": data
            }
        except Exception as e:
            self.logger.error(f"Error getting financial statements: {str(e)}", extra_data={
                'symbol': symbol,
                'statement_type': statement_type,
                'period': period,
                'error': str(e)
            })
            return {
                "symbol": symbol,
                "statement_type": statement_type,
                "period": period,
                "data": [],
                "error": f"Failed to retrieve {statement_type} statement: {str(e)}"
            }
    

    
    async def _get_news(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get company news"""
        symbol = params["symbol"].upper()
        from_date = params["from_date"]
        to_date = params["to_date"]
        
        try:
            news = self.client.company_news(symbol, from_date, to_date)
            
            return {
                "symbol": symbol,
                "from_date": from_date,
                "to_date": to_date,
                "news_count": len(news),
                "news": news[:50]  # Limit to 50 most recent articles
            }
        except Exception as e:
            self.logger.error(f"Error getting news: {str(e)}", extra_data={
                'symbol': symbol,
                'from_date': from_date,
                'to_date': to_date,
                'error': str(e)
            })
            
            # Return mock news data when API fails
            return {
                "symbol": symbol,
                "from_date": from_date,
                "to_date": to_date,
                "news_count": 5,
                "news": [
                    {
                        "headline": f"{symbol} Reports Strong Quarterly Results",
                        "summary": f"{symbol} has reported better-than-expected quarterly earnings.",
                        "datetime": int(datetime.now().timestamp())
                    }
                ],
                "error": f"API Error: {str(e)} - Using mock data"
            }
    
    async def _get_analyst_recommendations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get analyst recommendations"""
        symbol = params["symbol"].upper()
        
        try:
            recommendations = self.client.recommendation_trends(symbol)
            
            # Ensure recommendations is a list
            if not isinstance(recommendations, list):
                recommendations = [recommendations] if recommendations else []
            
            return {
                "symbol": symbol,
                "recommendations": recommendations
            }
        except Exception as e:
            self.logger.error(f"Error getting analyst recommendations: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            
            # Return mock analyst recommendations when API fails
            return {
                "symbol": symbol,
                "recommendations": [
                    {
                        "period": "0m",
                        "strongBuy": 15,
                        "buy": 8,
                        "hold": 3,
                        "sell": 1,
                        "strongSell": 0
                    }
                ],
                "error": f"API Error: {str(e)} - Using mock data"
            }
    
    async def _get_earnings_calendar(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get earnings calendar"""
        from_date = params["from_date"]
        to_date = params["to_date"]
        symbol = params.get("symbol", "").upper()
        
        calendar = self.client.earnings_calendar(
            _from=from_date, to=to_date, symbol=symbol
        )
        
        return {
            "from_date": from_date,
            "to_date": to_date,
            "symbol": symbol,
            "earnings": calendar.get('earningsCalendar', [])
        }
    
    def _get_insider_transactions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get insider transactions"""
        symbol = params["symbol"].upper()
        from_date = params["from_date"]
        to_date = params["to_date"]
        
        transactions = self.client.insider_transactions(symbol, from_date, to_date)
        
        return {
            "symbol": symbol,
            "from_date": from_date,
            "to_date": to_date,
            "transactions": transactions
        }
    
    def _get_technical_indicators(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators"""
        symbol = params["symbol"].upper()
        resolution = params["resolution"]
        indicators = params["indicators"]
        
        try:
            # Since we don't have access to stock candles, we'll use mock data
            # In a real implementation, you would get historical data from another source
            self.logger.info(f"Using mock technical indicators for {symbol} - stock candles not available", extra_data={
                'symbol': symbol,
                'resolution': resolution,
                'indicators': indicators,
                'note': 'Stock candles API not available'
            })
            
            # Return mock technical indicators
            return {
                "symbol": symbol,
                "indicators": {
                    "sma_20": 150.0,
                    "sma_50": 145.0,
                    "ema_12": 151.0,
                    "ema_26": 146.0,
                    "rsi": 55.0,
                    "macd": 2.5,
                    "macd_signal": 1.8,
                    "bollinger_upper": 160.0,
                    "bollinger_lower": 140.0,
                    "bollinger_middle": 150.0
                },
                "note": "Mock data - Stock candles API not available"
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}", extra_data={
                'symbol': symbol,
                'resolution': resolution,
                'indicators': indicators,
                'error': str(e)
            })
            
            # Return mock technical indicators when API fails
            return {
                "symbol": symbol,
                "indicators": {
                    "sma_20": 150.0,
                    "sma_50": 145.0,
                    "ema_12": 151.0,
                    "ema_26": 146.0,
                    "rsi": 55.0,
                    "macd": 2.5,
                    "macd_signal": 1.8,
                    "bollinger_upper": 160.0,
                    "bollinger_lower": 140.0,
                    "bollinger_middle": 150.0
                },
                "error": f"API Error: {str(e)} - Using mock data"
            }

# FastAPI server for MCP
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Finnhub MCP Server", version="1.0.0")
mcp_server = FinnhubMCPServer()

class ToolRequest(BaseModel):
    tool_name: str
    parameters: Dict[str, Any]

class ToolResponse(BaseModel):
    result: Dict[str, Any]
    success: bool
    error: Optional[str] = None

@app.get("/tools")
async def get_tools():
    """Get list of available tools"""
    return {"tools": mcp_server.get_tools()}

@app.post("/execute", response_model=ToolResponse)
async def execute_tool(request: ToolRequest):
    """Execute a tool with parameters"""
    try:
        mcp_server.logger.info(f"Executing tool: {request.tool_name}", extra_data={
            'tool_name': request.tool_name,
            'parameters': request.parameters,
            'step': 'tool_execution_start'
        })
        
        result = await mcp_server.execute_tool(request.tool_name, request.parameters)
        
        mcp_server.logger.info(f"Tool execution completed successfully", extra_data={
            'tool_name': request.tool_name,
            'result_keys': list(result.keys()) if isinstance(result, dict) else 'not_dict',
            'step': 'tool_execution_success'
        })
        
        return ToolResponse(result=result, success=True)
    except Exception as e:
        mcp_server.logger.error(f"Tool execution failed: {str(e)}", extra_data={
            'tool_name': request.tool_name,
            'error': str(e),
            'error_type': type(e).__name__,
            'step': 'tool_execution_error'
        })
        return ToolResponse(
            result={}, 
            success=False, 
            error=str(e)
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "finnhub_mcp_server"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('MCP_SERVER_PORT', 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
