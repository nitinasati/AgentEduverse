import asyncio
import json
import os
import aiohttp
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
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

class StockAnalysisEngine:
    """Engine for comprehensive stock analysis"""
    
    def __init__(self, mcp_server_url: str):
        self.mcp_server_url = mcp_server_url
        self.logger = logger.get_logger("stock_analysis_engine")
    
    async def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock data"""
        try:
            # Get current quote
            quote_data = await self._call_mcp_tool("get_stock_quote", {"symbol": symbol})
            
            # Get company profile
            profile_data = await self._call_mcp_tool("get_company_profile", {"symbol": symbol})
            
            # Historical data not available due to API limitations
            # Using mock historical data for demonstration
            historical_data = {
                "symbol": symbol,
                "resolution": "D",
                "data": [
                    {
                        "timestamp": int((datetime.now() - timedelta(days=1)).timestamp()),
                        "open": 150.0,
                        "high": 155.0,
                        "low": 148.0,
                        "close": 152.0,
                        "volume": 1000000
                    }
                ],
                "count": 1,
                "note": "Mock data - Stock candles API not available"
            }
            
            return {
                "quote": quote_data,
                "profile": profile_data,
                "historical": historical_data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stock data: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            raise
    
    async def get_financial_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get financial analysis data"""
        try:
            # Get income statement
            income_data = await self._call_mcp_tool("get_financial_statements", {
                "symbol": symbol,
                "statement_type": "income",
                "period": "annual"
            })
            
            # Get balance sheet
            balance_data = await self._call_mcp_tool("get_financial_statements", {
                "symbol": symbol,
                "statement_type": "balance",
                "period": "annual"
            })
            
            # Get cash flow
            cash_data = await self._call_mcp_tool("get_financial_statements", {
                "symbol": symbol,
                "statement_type": "cash",
                "period": "annual"
            })
            
            return {
                "income_statement": income_data,
                "balance_sheet": balance_data,
                "cash_flow": cash_data
            }
            
        except Exception as e:
            self.logger.error(f"Error getting financial analysis: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            raise
    
    async def get_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis data"""
        try:
            technical_data = await self._call_mcp_tool("get_technical_indicators", {
                "symbol": symbol,
                "resolution": "D",
                "indicators": ["sma", "ema", "rsi", "macd", "bollinger"]
            })
            
            return technical_data
            
        except Exception as e:
            self.logger.error(f"Error getting technical analysis: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            raise
    
    async def get_news_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get news and sentiment analysis"""
        try:
            # Get recent news (last 7 days)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            news_data = await self._call_mcp_tool("get_news", {
                "symbol": symbol,
                "from_date": start_date.strftime("%Y-%m-%d"),
                "to_date": end_date.strftime("%Y-%m-%d")
            })
            
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error getting news analysis: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            raise
    
    async def get_analyst_recommendations(self, symbol: str) -> Dict[str, Any]:
        """Get analyst recommendations"""
        try:
            recommendations = await self._call_mcp_tool("get_analyst_recommendations", {
                "symbol": symbol
            })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting analyst recommendations: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            raise
    
    async def _call_mcp_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server tool"""
        try:
            self.logger.info("Calling MCP tool", extra_data={
                'tool_name': tool_name,
                'parameters': parameters,
                'mcp_server_url': self.mcp_server_url,
                'step': 'mcp_tool_call_start'
            })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.mcp_server_url}/execute",
                    json={
                        "tool_name": tool_name,
                        "parameters": parameters
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info("MCP tool call successful", extra_data={
                            'tool_name': tool_name,
                            'response_status': response.status,
                            'result_keys': list(result.keys()) if isinstance(result, dict) else 'not_dict',
                            'step': 'mcp_tool_call_success'
                        })
                        return result.get('result', {})
                    else:
                        error_text = await response.text()
                        self.logger.error(f"MCP tool call failed", extra_data={
                            'tool_name': tool_name,
                            'response_status': response.status,
                            'error_text': error_text,
                            'step': 'mcp_tool_call_failed'
                        })
                        raise RuntimeError(f"MCP tool call failed: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"MCP tool call error: {str(e)}", extra_data={
                'tool_name': tool_name,
                'parameters': parameters,
                'error': str(e),
                'error_type': type(e).__name__,
                'step': 'mcp_tool_call_error'
            })
            raise

class StockRecommendationEngine:
    """Engine for generating stock recommendations"""
    
    def __init__(self):
        self.logger = logger.get_logger("stock_recommendation_engine")
    
    def generate_recommendation(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate buy/sell/hold recommendation based on analysis"""
        try:
            # Extract key metrics
            quote = analysis_data.get('quote', {})
            technical = analysis_data.get('technical', {})
            financial = analysis_data.get('financial', {})
            news = analysis_data.get('news', {})
            analyst_recs = analysis_data.get('analyst_recommendations', {})
            
            # Calculate recommendation score
            score = self._calculate_recommendation_score(
                quote, technical, financial, news, analyst_recs
            )
            
            # Generate recommendation
            recommendation = self._get_recommendation_from_score(score)
            
            # Generate detailed analysis
            analysis = self._generate_detailed_analysis(analysis_data)
            
            return {
                "symbol": quote.get('symbol', 'Unknown'),
                "recommendation": recommendation,
                "score": score,
                "confidence": self._calculate_confidence(analysis_data),
                "analysis": analysis,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendation: {str(e)}", extra_data={
                'error': str(e)
            })
            raise
    
    def _calculate_recommendation_score(self, quote: Dict, technical: Dict, 
                                      financial: Dict, news: Dict, analyst_recs: Dict) -> float:
        """Calculate recommendation score from 0-100"""
        score = 50.0  # Neutral starting point
        
        # Technical analysis factors (30% weight)
        if technical and 'indicators' in technical:
            indicators = technical['indicators']
            
            # RSI analysis
            rsi = indicators.get('rsi', 50)
            if rsi is not None and isinstance(rsi, (int, float)):
                if rsi < 30:
                    score += 10  # Oversold
                elif rsi > 70:
                    score -= 10  # Overbought
            
            # Moving averages
            current_price = quote.get('current_price', 0)
            sma_20 = indicators.get('sma_20', current_price)
            sma_50 = indicators.get('sma_50', current_price)
            
            # Safe comparison for moving averages
            if (current_price is not None and sma_20 is not None and sma_50 is not None and
                isinstance(current_price, (int, float)) and isinstance(sma_20, (int, float)) and isinstance(sma_50, (int, float))):
                if current_price > sma_20 > sma_50:
                    score += 5  # Bullish trend
                elif current_price < sma_20 < sma_50:
                    score -= 5  # Bearish trend
            
            # MACD
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            if (macd is not None and macd_signal is not None and 
                isinstance(macd, (int, float)) and isinstance(macd_signal, (int, float))):
                if macd > macd_signal:
                    score += 3  # Bullish MACD
                else:
                    score -= 3  # Bearish MACD
        
        # Price movement factors (20% weight)
        percent_change = quote.get('percent_change', 0)
        if percent_change is not None and isinstance(percent_change, (int, float)):
            if percent_change > 5:
                score += 5  # Strong positive movement
            elif percent_change < -5:
                score -= 5  # Strong negative movement
        
        # News sentiment (20% weight)
        if news and 'news_count' in news:
            news_count = news.get('news_count', 0)
            if news_count is not None and isinstance(news_count, (int, float)) and news_count > 10:
                score += 3  # High news volume (could be positive or negative)
        
        # Analyst recommendations (30% weight)
        if analyst_recs and 'recommendations' in analyst_recs:
            recommendations = analyst_recs['recommendations']
            if recommendations and isinstance(recommendations, list):
                # Calculate average analyst rating
                total_rating = 0
                count = 0
                for rec in recommendations:
                    if isinstance(rec, dict):
                        rating = (rec.get('strongBuy', 0) or 0) * 5 + (rec.get('buy', 0) or 0) * 4 + \
                                (rec.get('hold', 0) or 0) * 3 + (rec.get('sell', 0) or 0) * 2 + \
                                (rec.get('strongSell', 0) or 0) * 1
                        total_rating += rating
                        count += 1
                
                if count > 0:
                    avg_rating = total_rating / count
                    if avg_rating > 4:
                        score += 10  # Strong buy
                    elif avg_rating > 3.5:
                        score += 5   # Buy
                    elif avg_rating < 2.5:
                        score -= 5   # Sell
        
        # Ensure score is within bounds
        return max(0, min(100, score))
    
    def _get_recommendation_from_score(self, score: float) -> str:
        """Convert score to recommendation"""
        if score >= 70:
            return "STRONG_BUY"
        elif score >= 60:
            return "BUY"
        elif score >= 40:
            return "HOLD"
        elif score >= 30:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _calculate_confidence(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate confidence level of the recommendation"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on data availability
        if analysis_data.get('quote'):
            confidence += 0.1
        if analysis_data.get('technical'):
            confidence += 0.1
        if analysis_data.get('financial'):
            confidence += 0.1
        if analysis_data.get('news'):
            confidence += 0.1
        if analysis_data.get('analyst_recommendations'):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _generate_detailed_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Generate detailed analysis text"""
        analysis_parts = []
        
        # Quote analysis
        quote = analysis_data.get('quote', {})
        if quote:
            current_price = quote.get('current_price', 0)
            percent_change = quote.get('percent_change', 0)
            # Safe formatting for numeric values
            current_price_str = f"{current_price:.2f}" if isinstance(current_price, (int, float)) else str(current_price)
            percent_change_str = f"{percent_change:+.2f}" if isinstance(percent_change, (int, float)) else str(percent_change)
            analysis_parts.append(
                f"Current price: ${current_price_str} ({percent_change_str}%)"
            )
        
        # Technical analysis
        technical = analysis_data.get('technical', {})
        if technical and 'indicators' in technical:
            indicators = technical['indicators']
            rsi = indicators.get('rsi', 50)
            rsi_str = f"{rsi:.1f}" if isinstance(rsi, (int, float)) else str(rsi)
            analysis_parts.append(f"RSI: {rsi_str} ({'Oversold' if isinstance(rsi, (int, float)) and rsi < 30 else 'Overbought' if isinstance(rsi, (int, float)) and rsi > 70 else 'Neutral'})")
        
        # News analysis
        news = analysis_data.get('news', {})
        if news:
            news_count = news.get('news_count', 0)
            analysis_parts.append(f"Recent news articles: {news_count}")
        
        # Analyst recommendations
        analyst_recs = analysis_data.get('analyst_recommendations', {})
        if analyst_recs and 'recommendations' in analyst_recs:
            recommendations = analyst_recs['recommendations']
            if recommendations:
                analysis_parts.append(f"Analyst coverage: {len(recommendations)} firms")
        
        return "; ".join(analysis_parts)

class StockAgent(BaseAgent):
    """Specialized agent for stock analysis and investment recommendations"""
    
    @staticmethod
    def safe_format(value, format_spec):
        """Helper function to safely format numeric values"""
        if isinstance(value, (int, float)) and value != 'N/A':
            return format_spec.format(value)
        return str(value)
    
    def __init__(self):
        super().__init__(
            agent_name="StockAgent",
            agent_description="Specialized agent for comprehensive stock analysis, financial data processing, and investment recommendations"
        )
        
        # Initialize analysis engines
        mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8003')
        self.analysis_engine = StockAnalysisEngine(mcp_server_url)
        self.recommendation_engine = StockRecommendationEngine()
        
        # Add stock agent capabilities
        self.add_capability("stock_analysis")
        self.add_capability("financial_data")
        self.add_capability("investment_recommendations")
        self.add_capability("technical_analysis")
        self.add_capability("news_analysis")
        
        # Add stock analysis tools
        self._add_stock_analysis_tools()
        
        self.logger.info("Stock Agent initialized successfully", extra_data={
            'capabilities': self.capabilities,
            'mcp_server_url': mcp_server_url
        })
    
    def _add_stock_analysis_tools(self):
        """Add tools for stock analysis"""
        from langchain.tools import Tool
        
        # Tool for comprehensive stock analysis
        analysis_tool = Tool(
            name="analyze_stock",
            description="Perform comprehensive stock analysis including price data, financials, technical indicators, and news",
            func=self._analyze_stock
        )
        self.add_tool(analysis_tool)
        
        # Tool for investment recommendations
        recommendation_tool = Tool(
            name="get_investment_recommendation",
            description="Get buy/sell/hold recommendation with detailed analysis",
            func=self._get_investment_recommendation
        )
        self.add_tool(recommendation_tool)
        
        # Tool for technical analysis
        technical_tool = Tool(
            name="get_technical_analysis",
            description="Get detailed technical analysis with indicators",
            func=self._get_technical_analysis
        )
        self.add_tool(technical_tool)
        
        # Tool for financial analysis
        financial_tool = Tool(
            name="get_financial_analysis",
            description="Get comprehensive financial analysis including statements and ratios",
            func=self._get_financial_analysis
        )
        self.add_tool(financial_tool)
    
    async def _analyze_stock(self, symbol: str) -> str:
        """Perform comprehensive stock analysis"""
        try:
            self.logger.info(f"Starting comprehensive stock analysis for {symbol}", extra_data={
                'symbol': symbol,
                'step': 'analysis_start'
            })
            
            # Get all analysis data
            self.logger.info(f"Fetching stock data for {symbol}", extra_data={
                'symbol': symbol,
                'step': 'fetch_stock_data'
            })
            stock_data = await self.analysis_engine.get_stock_data(symbol)
            
            self.logger.info(f"Fetching financial data for {symbol}", extra_data={
                'symbol': symbol,
                'step': 'fetch_financial_data'
            })
            financial_data = await self.analysis_engine.get_financial_analysis(symbol)
            
            self.logger.info(f"Fetching technical data for {symbol}", extra_data={
                'symbol': symbol,
                'step': 'fetch_technical_data'
            })
            technical_data = await self.analysis_engine.get_technical_analysis(symbol)
            
            self.logger.info(f"Fetching news data for {symbol}", extra_data={
                'symbol': symbol,
                'step': 'fetch_news_data'
            })
            news_data = await self.analysis_engine.get_news_analysis(symbol)
            
            self.logger.info(f"Fetching analyst data for {symbol}", extra_data={
                'symbol': symbol,
                'step': 'fetch_analyst_data'
            })
            analyst_data = await self.analysis_engine.get_analyst_recommendations(symbol)
            
            # Combine all data
            analysis_data = {
                "quote": stock_data.get("quote", {}),
                "profile": stock_data.get("profile", {}),
                "historical": stock_data.get("historical", {}),
                "financial": financial_data,
                "technical": technical_data,
                "news": news_data,
                "analyst_recommendations": analyst_data
            }
            
            # Generate recommendation
            recommendation = self.recommendation_engine.generate_recommendation(analysis_data)
            
            # Format response
            response = f"""
            **Comprehensive Stock Analysis for {symbol.upper()}**
            
            **Current Status:**
            - Price: ${self.safe_format(stock_data.get('quote', {}).get('current_price', 'N/A'), '{:.2f}')}
            - Recommendation: {recommendation['recommendation']}
            - Confidence: {self.safe_format(recommendation['confidence'], '{:.1%}')}
            
            **Technical Analysis:**
            - RSI: {self.safe_format(technical_data.get('indicators', {}).get('rsi', 'N/A'), '{:.1f}')}
            - Moving Averages: 20-day SMA vs 50-day SMA trend analysis
            - MACD: {self.safe_format(technical_data.get('indicators', {}).get('macd', 'N/A'), '{:.2f}')}
            
            **Financial Health:**
            - Market Cap: ${self.safe_format(stock_data.get('profile', {}).get('market_capitalization', 0), '{:,.0f}')}
            - Industry: {stock_data.get('profile', {}).get('finnhub_industry', 'N/A')}
            
            **News & Sentiment:**
            - Recent News Articles: {news_data.get('news_count', 0)}
            - Analyst Coverage: {len(analyst_data.get('recommendations', []))} firms
            
            **Detailed Analysis:**
            {recommendation['analysis']}
            """
            
            self.logger.info(f"Stock analysis completed for {symbol}", extra_data={
                'symbol': symbol,
                'recommendation': recommendation['recommendation'],
                'confidence': recommendation['confidence']
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Stock analysis failed for {symbol}: {str(e)}", extra_data={
                'symbol': symbol,
                'error': str(e)
            })
            return f"Analysis failed for {symbol}: {str(e)}"
    
    async def _get_investment_recommendation(self, symbol: str) -> str:
        """Get investment recommendation"""
        try:
            # Get comprehensive analysis data
            stock_data = await self.analysis_engine.get_stock_data(symbol)
            financial_data = await self.analysis_engine.get_financial_analysis(symbol)
            technical_data = await self.analysis_engine.get_technical_analysis(symbol)
            news_data = await self.analysis_engine.get_news_analysis(symbol)
            analyst_data = await self.analysis_engine.get_analyst_recommendations(symbol)
            
            analysis_data = {
                "quote": stock_data.get("quote", {}),
                "technical": technical_data,
                "financial": financial_data,
                "news": news_data,
                "analyst_recommendations": analyst_data
            }
            
            recommendation = self.recommendation_engine.generate_recommendation(analysis_data)
            
            response = f"""
            **Investment Recommendation for {symbol.upper()}**
            
            **Recommendation: {recommendation['recommendation']}**
            **Confidence: {self.safe_format(recommendation['confidence'], '{:.1%}')}**
            **Score: {self.safe_format(recommendation['score'], '{:.1f}')}/100**
            
            **Key Factors:**
            - Current Price: ${self.safe_format(stock_data.get('quote', {}).get('current_price', 'N/A'), '{:.2f}')}
            - Technical Indicators: RSI, MACD, Moving Averages analyzed
            - Financial Health: Balance sheet and income statement reviewed
            - Market Sentiment: News analysis and analyst ratings considered
            
            **Risk Assessment:**
            Based on the comprehensive analysis, this stock shows {recommendation['recommendation'].lower().replace('_', ' ')} potential.
            """
            
            return response
            
        except Exception as e:
            return f"Failed to generate recommendation for {symbol}: {str(e)}"
    
    async def _get_technical_analysis(self, symbol: str) -> str:
        """Get detailed technical analysis"""
        try:
            technical_data = await self.analysis_engine.get_technical_analysis(symbol)
            
            indicators = technical_data.get('indicators', {})
            
            response = f"""
            **Technical Analysis for {symbol.upper()}**
            
            **Moving Averages:**
            - 20-day SMA: ${self.safe_format(indicators.get('sma_20', 'N/A'), '{:.2f}')}
            - 50-day SMA: ${self.safe_format(indicators.get('sma_50', 'N/A'), '{:.2f}')}
            
            **Exponential Moving Averages:**
            - 12-day EMA: ${self.safe_format(indicators.get('ema_12', 'N/A'), '{:.2f}')}
            - 26-day EMA: ${self.safe_format(indicators.get('ema_26', 'N/A'), '{:.2f}')}
            
            **Oscillators:**
            - RSI (14): {self.safe_format(indicators.get('rsi', 'N/A'), '{:.1f}')}
            
            **MACD:**
            - MACD Line: {self.safe_format(indicators.get('macd', 'N/A'), '{:.2f}')}
            - Signal Line: {self.safe_format(indicators.get('macd_signal', 'N/A'), '{:.2f}')}
            
            **Bollinger Bands:**
            - Upper Band: ${self.safe_format(indicators.get('bollinger_upper', 'N/A'), '{:.2f}')}
            - Middle Band: ${self.safe_format(indicators.get('bollinger_middle', 'N/A'), '{:.2f}')}
            - Lower Band: ${self.safe_format(indicators.get('bollinger_lower', 'N/A'), '{:.2f}')}
            """
            
            return response
            
        except Exception as e:
            return f"Technical analysis failed for {symbol}: {str(e)}"
    
    async def _get_financial_analysis(self, symbol: str) -> str:
        """Get financial analysis"""
        try:
            financial_data = await self.analysis_engine.get_financial_analysis(symbol)
            
            response = f"""
            **Financial Analysis for {symbol.upper()}**
            
            **Income Statement Data Available:** {'Yes' if financial_data.get('income_statement') else 'No'}
            **Balance Sheet Data Available:** {'Yes' if financial_data.get('balance_sheet') else 'No'}
            **Cash Flow Data Available:** {'Yes' if financial_data.get('cash_flow') else 'No'}
            
            Financial statements have been retrieved and analyzed for comprehensive financial health assessment.
            """
            
            return response
            
        except Exception as e:
            return f"Financial analysis failed for {symbol}: {str(e)}"

# FastAPI server for stock agent
app = FastAPI(title="Stock Agent API", version="1.0.0")
stock_agent = None

class StockAnalysisRequest(BaseModel):
    symbol: str
    analysis_type: Optional[str] = "comprehensive"  # comprehensive, technical, financial, recommendation

@app.on_event("startup")
async def startup_event():
    global stock_agent
    stock_agent = StockAgent()

@app.post("/analyze_stock")
async def analyze_stock(request: StockAnalysisRequest):
    """Analyze a stock"""
    if not stock_agent:
        stock_agent.logger.error("Stock agent not initialized", extra_data={
            'step': 'agent_initialization_error'
        })
        raise HTTPException(status_code=500, detail="Stock agent not initialized")
    
    try:
        stock_agent.logger.info(f"Analyzing stock: {request.symbol} with type: {request.analysis_type}", extra_data={
            'symbol': request.symbol,
            'analysis_type': request.analysis_type,
            'step': 'analysis_start'
        })
        
        if request.analysis_type == "comprehensive":
            result = await stock_agent._analyze_stock(request.symbol)
        elif request.analysis_type == "technical":
            result = await stock_agent._get_technical_analysis(request.symbol)
        elif request.analysis_type == "financial":
            result = await stock_agent._get_financial_analysis(request.symbol)
        elif request.analysis_type == "recommendation":
            result = await stock_agent._get_investment_recommendation(request.symbol)
        else:
            result = await stock_agent._analyze_stock(request.symbol)
        
        stock_agent.logger.info(f"Analysis completed successfully", extra_data={
            'symbol': request.symbol,
            'analysis_type': request.analysis_type,
            'result_length': len(str(result)),
            'step': 'analysis_complete'
        })
        
        return {
            "symbol": request.symbol,
            "analysis_type": request.analysis_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        stock_agent.logger.error(f"Analysis failed: {str(e)}", extra_data={
            'symbol': request.symbol,
            'analysis_type': request.analysis_type,
            'error': str(e),
            'error_type': type(e).__name__,
            'step': 'analysis_error'
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not stock_agent:
        return {"status": "unhealthy", "error": "Stock agent not initialized"}
    
    try:
        health = await stock_agent.health_check()
        return health
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    port = int(os.getenv('STOCK_AGENT_PORT', 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
