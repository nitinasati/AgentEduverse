#!/usr/bin/env python3
"""
AgentEduverse System Test Script

This script tests all components of the AgentEduverse system:
1. MCP Server (Finnhub API integration)
2. Master Agent (coordination and discovery)
3. Stock Agent (stock analysis and recommendations)
4. Streamlit Frontend (user interface)

Usage:
    python test_system.py [--component COMPONENT]
"""

import os
import sys
import asyncio
import aiohttp
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import logger

class SystemTester:
    """Test all components of the AgentEduverse system"""
    
    def __init__(self):
        self.logger = logger.get_logger("system_tester")
        
        # Service URLs
        self.mcp_server_url = os.getenv('MCP_SERVER_URL', 'http://localhost:8003')
        self.master_agent_url = os.getenv('MASTER_AGENT_URL', 'http://localhost:8001')
        self.stock_agent_url = os.getenv('STOCK_AGENT_URL', 'http://localhost:8002')
        self.frontend_url = os.getenv('STREAMLIT_SERVER_URL', 'http://localhost:8501')
        
        # Test results
        self.test_results = {}
    
    async def test_mcp_server(self) -> Dict[str, Any]:
        """Test MCP server functionality"""
        self.logger.info("Testing MCP server")
        print("🔧 Testing MCP Server...")
        
        results = {
            'health_check': False,
            'tools_list': False,
            'tool_execution': False,
            'errors': []
        }
        
        try:
            # Test health check
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mcp_server_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        results['health_check'] = health_data.get('status') == 'healthy'
                        print(f"   ✅ Health check: {results['health_check']}")
                    else:
                        results['errors'].append(f"Health check failed: {response.status}")
                        print(f"   ❌ Health check failed: {response.status}")
            
            # Test tools list
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.mcp_server_url}/tools") as response:
                    if response.status == 200:
                        tools_data = await response.json()
                        tools = tools_data.get('tools', [])
                        results['tools_list'] = len(tools) > 0
                        print(f"   ✅ Tools list: {len(tools)} tools available")
                    else:
                        results['errors'].append(f"Tools list failed: {response.status}")
                        print(f"   ❌ Tools list failed: {response.status}")
            
            # Test tool execution (stock quote)
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.mcp_server_url}/execute",
                    json={
                        "tool_name": "get_stock_quote",
                        "parameters": {"symbol": "AAPL"}
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['tool_execution'] = result.get('success', False)
                        print(f"   ✅ Tool execution: {results['tool_execution']}")
                    else:
                        results['errors'].append(f"Tool execution failed: {response.status}")
                        print(f"   ❌ Tool execution failed: {response.status}")
        
        except Exception as e:
            error_msg = f"MCP server test error: {str(e)}"
            results['errors'].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        return results
    
    async def test_master_agent(self) -> Dict[str, Any]:
        """Test master agent functionality"""
        self.logger.info("Testing master agent")
        print("🤖 Testing Master Agent...")
        
        results = {
            'health_check': False,
            'agent_registration': False,
            'task_processing': False,
            'agent_discovery': False,
            'errors': []
        }
        
        try:
            # Test health check
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.master_agent_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        results['health_check'] = health_data.get('status') == 'healthy'
                        print(f"   ✅ Health check: {results['health_check']}")
                    else:
                        results['errors'].append(f"Health check failed: {response.status}")
                        print(f"   ❌ Health check failed: {response.status}")
            
            # Test agent registration
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.master_agent_url}/register_agent",
                    json={
                        "name": "TestAgent",
                        "description": "Test agent for system testing",
                        "capabilities": ["test_capability"],
                        "endpoint": "http://localhost:9999"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['agent_registration'] = 'agent_id' in result
                        print(f"   ✅ Agent registration: {results['agent_registration']}")
                    else:
                        results['errors'].append(f"Agent registration failed: {response.status}")
                        print(f"   ❌ Agent registration failed: {response.status}")
            
            # Test task processing
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.master_agent_url}/process_task",
                    json={
                        "task": "Hello, this is a test message",
                        "session_id": "test_session"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['task_processing'] = 'response' in result
                        print(f"   ✅ Task processing: {results['task_processing']}")
                    else:
                        results['errors'].append(f"Task processing failed: {response.status}")
                        print(f"   ❌ Task processing failed: {response.status}")
            
            # Test agent discovery
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.master_agent_url}/agents") as response:
                    if response.status == 200:
                        agents_data = await response.json()
                        agents = agents_data.get('agents', [])
                        results['agent_discovery'] = len(agents) > 0
                        print(f"   ✅ Agent discovery: {len(agents)} agents found")
                    else:
                        results['errors'].append(f"Agent discovery failed: {response.status}")
                        print(f"   ❌ Agent discovery failed: {response.status}")
        
        except Exception as e:
            error_msg = f"Master agent test error: {str(e)}"
            results['errors'].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        return results
    
    async def test_stock_agent(self) -> Dict[str, Any]:
        """Test stock agent functionality"""
        self.logger.info("Testing stock agent")
        print("📈 Testing Stock Agent...")
        
        results = {
            'health_check': False,
            'stock_analysis': False,
            'technical_analysis': False,
            'financial_analysis': False,
            'errors': []
        }
        
        try:
            # Test health check
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.stock_agent_url}/health") as response:
                    if response.status == 200:
                        health_data = await response.json()
                        results['health_check'] = health_data.get('status') == 'healthy'
                        print(f"   ✅ Health check: {results['health_check']}")
                    else:
                        results['errors'].append(f"Health check failed: {response.status}")
                        print(f"   ❌ Health check failed: {response.status}")
            
            # Test comprehensive stock analysis
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.stock_agent_url}/analyze_stock",
                    json={
                        "symbol": "AAPL",
                        "analysis_type": "comprehensive"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['stock_analysis'] = 'result' in result
                        print(f"   ✅ Stock analysis: {results['stock_analysis']}")
                    else:
                        results['errors'].append(f"Stock analysis failed: {response.status}")
                        print(f"   ❌ Stock analysis failed: {response.status}")
            
            # Test technical analysis
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.stock_agent_url}/analyze_stock",
                    json={
                        "symbol": "MSFT",
                        "analysis_type": "technical"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['technical_analysis'] = 'result' in result
                        print(f"   ✅ Technical analysis: {results['technical_analysis']}")
                    else:
                        results['errors'].append(f"Technical analysis failed: {response.status}")
                        print(f"   ❌ Technical analysis failed: {response.status}")
            
            # Test financial analysis
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.stock_agent_url}/analyze_stock",
                    json={
                        "symbol": "GOOGL",
                        "analysis_type": "financial"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['financial_analysis'] = 'result' in result
                        print(f"   ✅ Financial analysis: {results['financial_analysis']}")
                    else:
                        results['errors'].append(f"Financial analysis failed: {response.status}")
                        print(f"   ❌ Financial analysis failed: {response.status}")
        
        except Exception as e:
            error_msg = f"Stock agent test error: {str(e)}"
            results['errors'].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        return results
    
    async def test_frontend(self) -> Dict[str, Any]:
        """Test frontend accessibility"""
        self.logger.info("Testing frontend")
        print("📊 Testing Frontend...")
        
        results = {
            'accessibility': False,
            'errors': []
        }
        
        try:
            # Test frontend accessibility
            async with aiohttp.ClientSession() as session:
                async with session.get(self.frontend_url, timeout=10) as response:
                    if response.status == 200:
                        results['accessibility'] = True
                        print(f"   ✅ Frontend accessible: {results['accessibility']}")
                    else:
                        results['errors'].append(f"Frontend not accessible: {response.status}")
                        print(f"   ❌ Frontend not accessible: {response.status}")
        
        except Exception as e:
            error_msg = f"Frontend test error: {str(e)}"
            results['errors'].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        return results
    
    async def test_integration(self) -> Dict[str, Any]:
        """Test integration between components"""
        self.logger.info("Testing system integration")
        print("🔗 Testing System Integration...")
        
        results = {
            'master_to_stock': False,
            'master_to_mcp': False,
            'end_to_end': False,
            'errors': []
        }
        
        try:
            # Test master agent to stock agent communication
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.master_agent_url}/process_task",
                    json={
                        "task": "Analyze Apple stock and provide recommendation",
                        "session_id": "integration_test"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['master_to_stock'] = 'response' in result
                        print(f"   ✅ Master to Stock communication: {results['master_to_stock']}")
                    else:
                        results['errors'].append(f"Master to Stock communication failed: {response.status}")
                        print(f"   ❌ Master to Stock communication failed: {response.status}")
            
            # Test master agent to MCP communication
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.master_agent_url}/process_task",
                    json={
                        "task": "Get stock quote for Tesla",
                        "session_id": "integration_test"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['master_to_mcp'] = 'response' in result
                        print(f"   ✅ Master to MCP communication: {results['master_to_mcp']}")
                    else:
                        results['errors'].append(f"Master to MCP communication failed: {response.status}")
                        print(f"   ❌ Master to MCP communication failed: {response.status}")
            
            # Test end-to-end workflow
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.master_agent_url}/process_task",
                    json={
                        "task": "Provide comprehensive analysis and recommendation for Microsoft stock",
                        "session_id": "integration_test"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        results['end_to_end'] = 'response' in result
                        print(f"   ✅ End-to-end workflow: {results['end_to_end']}")
                    else:
                        results['errors'].append(f"End-to-end workflow failed: {response.status}")
                        print(f"   ❌ End-to-end workflow failed: {response.status}")
        
        except Exception as e:
            error_msg = f"Integration test error: {str(e)}"
            results['errors'].append(error_msg)
            print(f"   ❌ {error_msg}")
        
        return results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all system tests"""
        self.logger.info("Starting comprehensive system tests")
        print("🧪 Starting Comprehensive System Tests...")
        print("=" * 60)
        
        # Run individual component tests
        self.test_results['mcp_server'] = await self.test_mcp_server()
        print()
        
        self.test_results['master_agent'] = await self.test_master_agent()
        print()
        
        self.test_results['stock_agent'] = await self.test_stock_agent()
        print()
        
        self.test_results['frontend'] = await self.test_frontend()
        print()
        
        # Run integration tests
        self.test_results['integration'] = await self.test_integration()
        print()
        
        return self.test_results
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report"""
        report = []
        report.append("=" * 60)
        report.append("🧪 AGENTEDUVERSE SYSTEM TEST REPORT")
        report.append("=" * 60)
        report.append(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Component test results
        for component, results in self.test_results.items():
            report.append(f"📋 {component.upper().replace('_', ' ')}")
            report.append("-" * 40)
            
            # Count passed tests
            passed_tests = sum(1 for key, value in results.items() 
                             if key != 'errors' and isinstance(value, bool) and value)
            total_tests = sum(1 for key, value in results.items() 
                            if key != 'errors' and isinstance(value, bool))
            
            report.append(f"Tests Passed: {passed_tests}/{total_tests}")
            
            # Individual test results
            for key, value in results.items():
                if key != 'errors' and isinstance(value, bool):
                    status = "✅ PASS" if value else "❌ FAIL"
                    report.append(f"  {key}: {status}")
            
            # Errors
            if results.get('errors'):
                report.append("  Errors:")
                for error in results['errors']:
                    report.append(f"    - {error}")
            
            report.append("")
        
        # Overall summary
        report.append("📊 OVERALL SUMMARY")
        report.append("-" * 40)
        
        total_passed = 0
        total_tests = 0
        total_errors = 0
        
        for component, results in self.test_results.items():
            for key, value in results.items():
                if key != 'errors' and isinstance(value, bool):
                    total_tests += 1
                    if value:
                        total_passed += 1
            
            total_errors += len(results.get('errors', []))
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Passed: {total_passed}")
        report.append(f"Failed: {total_tests - total_passed}")
        report.append(f"Success Rate: {success_rate:.1f}%")
        report.append(f"Total Errors: {total_errors}")
        
        if success_rate >= 80:
            report.append("🎉 Overall Status: EXCELLENT")
        elif success_rate >= 60:
            report.append("✅ Overall Status: GOOD")
        elif success_rate >= 40:
            report.append("⚠️  Overall Status: FAIR")
        else:
            report.append("❌ Overall Status: POOR")
        
        report.append("=" * 60)
        
        return "\n".join(report)

async def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="AgentEduverse System Test Script")
    parser.add_argument("--component", choices=['mcp', 'master', 'stock', 'frontend', 'integration'], 
                       help="Test specific component only")
    parser.add_argument("--output", help="Output test report to file")
    
    args = parser.parse_args()
    
    # Create tester
    tester = SystemTester()
    
    try:
        if args.component:
            # Test specific component
            if args.component == 'mcp':
                results = await tester.test_mcp_server()
            elif args.component == 'master':
                results = await tester.test_master_agent()
            elif args.component == 'stock':
                results = await tester.test_stock_agent()
            elif args.component == 'frontend':
                results = await tester.test_frontend()
            elif args.component == 'integration':
                results = await tester.test_integration()
            
            tester.test_results[args.component] = results
        else:
            # Run all tests
            await tester.run_all_tests()
        
        # Generate and display report
        report = tester.generate_report()
        print(report)
        
        # Save report to file if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"\n📄 Test report saved to: {args.output}")
        
        # Exit with appropriate code
        total_passed = 0
        total_tests = 0
        
        for component, results in tester.test_results.items():
            for key, value in results.items():
                if key != 'errors' and isinstance(value, bool):
                    total_tests += 1
                    if value:
                        total_passed += 1
        
        success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        if success_rate >= 60:
            print("\n🎉 Tests completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Tests failed!")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Testing failed with error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
