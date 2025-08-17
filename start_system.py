#!/usr/bin/env python3
"""
AgentEduverse System Startup Script

This script starts all components of the AgentEduverse system:
1. MCP Server (Finnhub API integration)
2. Master Agent (coordination and discovery)
3. Stock Agent (stock analysis and recommendations)
4. Streamlit Frontend (user interface)

Usage:
    python start_system.py [--mcp-only] [--agents-only] [--frontend-only]
"""

import os
import sys
import time
import subprocess
import signal
import argparse
from pathlib import Path
import asyncio
import aiohttp
import threading
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.logger import logger

class SystemManager:
    """Manages the startup and shutdown of all system components"""
    
    def __init__(self):
        self.processes = {}
        self.logger = logger.get_logger("system_manager")
        
        # Component configurations
        self.components = {
            'mcp_server': {
                'script': 'mcp_server/finnhub_server.py',
                'port': int(os.getenv('MCP_SERVER_PORT', 8003)),
                'name': 'MCP Server',
                'env_vars': {
                    'FINNHUB_API_KEY': os.getenv('FINNHUB_API_KEY'),
                    'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
                }
            },
            'master_agent': {
                'script': 'agents/master_agent.py',
                'port': int(os.getenv('MASTER_AGENT_PORT', 8001)),
                'name': 'Master Agent',
                'env_vars': {
                    'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
                    'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
                    'AWS_REGION': os.getenv('AWS_REGION', 'us-east-1'),
                    'LANGCHAIN_API_KEY': os.getenv('LANGCHAIN_API_KEY'),
                    'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
                }
            },
            'stock_agent': {
                'script': 'agents/stock_agent.py',
                'port': int(os.getenv('STOCK_AGENT_PORT', 8002)),
                'name': 'Stock Agent',
                'env_vars': {
                    'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
                    'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
                    'AWS_REGION': os.getenv('AWS_REGION', 'us-east-1'),
                    'LANGCHAIN_API_KEY': os.getenv('LANGCHAIN_API_KEY'),
                    'MCP_SERVER_URL': os.getenv('MCP_SERVER_URL', 'http://localhost:8003'),
                    'LOG_LEVEL': os.getenv('LOG_LEVEL', 'INFO')
                }
            },
            'frontend': {
                'script': 'frontend/app.py',
                'port': int(os.getenv('STREAMLIT_SERVER_PORT', 8501)),
                'name': 'Streamlit Frontend',
                'env_vars': {
                    'MASTER_AGENT_URL': os.getenv('MASTER_AGENT_URL', 'http://localhost:8001'),
                    'STOCK_AGENT_URL': os.getenv('STOCK_AGENT_URL', 'http://localhost:8002'),
                    'MCP_SERVER_URL': os.getenv('MCP_SERVER_URL', 'http://localhost:8003')
                }
            }
        }
    
    def check_environment(self) -> bool:
        """Check if required environment variables are set"""
        required_vars = [
            'FINNHUB_API_KEY',
            'AWS_ACCESS_KEY_ID',
            'AWS_SECRET_ACCESS_KEY',
            'AWS_REGION'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
            print(f"❌ Error: Missing required environment variables: {missing_vars}")
            print("Please set these variables in your environment or .env file")
            return False
        
        self.logger.info("Environment check passed")
        return True
    
    def start_component(self, component_name: str) -> bool:
        """Start a specific component"""
        if component_name not in self.components:
            self.logger.error(f"Unknown component: {component_name}")
            return False
        
        config = self.components[component_name]
        script_path = Path(config['script'])
        
        if not script_path.exists():
            self.logger.error(f"Script not found: {script_path}")
            print(f"❌ Error: Script not found: {script_path}")
            return False
        
        try:
            # Set environment variables
            env = os.environ.copy()
            env.update(config['env_vars'])
            
            # Start the process
            if component_name == 'frontend':
                # Use streamlit for frontend
                cmd = ['streamlit', 'run', str(script_path), '--server.port', str(config['port'])]
            else:
                # Use python for other components
                cmd = [sys.executable, str(script_path)]
            
            self.logger.info(f"Starting {config['name']} on port {config['port']}")
            print(f"🚀 Starting {config['name']} on port {config['port']}...")
            
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Start a thread to read and display process output
            def read_output(process, component_name):
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            print(f"[{component_name.upper()}] {line.rstrip()}")
                except Exception as e:
                    print(f"Error reading output from {component_name}: {e}")
            
            output_thread = threading.Thread(
                target=read_output, 
                args=(process, component_name),
                daemon=True
            )
            output_thread.start()
            
            self.processes[component_name] = {
                'process': process,
                'config': config,
                'started_at': time.time(),
                'output_thread': output_thread
            }
            
            # Wait a bit for the process to start
            time.sleep(2)
            
            if process.poll() is None:
                self.logger.info(f"{config['name']} started successfully")
                print(f"✅ {config['name']} started successfully")
                return True
            else:
                # Process failed to start
                self.logger.error(f"{config['name']} failed to start")
                print(f"❌ {config['name']} failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting {config['name']}: {str(e)}")
            print(f"❌ Error starting {config['name']}: {str(e)}")
            return False
    
    async def check_component_health(self, component_name: str) -> bool:
        """Check if a component is healthy"""
        if component_name not in self.components:
            return False
        
        config = self.components[component_name]
        health_url = f"http://localhost:{config['port']}/health"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    if response.status == 200:
                        return True
                    else:
                        return False
        except Exception:
            return False
    
    async def wait_for_component(self, component_name: str, timeout: int = 30) -> bool:
        """Wait for a component to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if await self.check_component_health(component_name):
                return True
            await asyncio.sleep(1)
        
        return False
    
    def start_system(self, components: List[str] = None) -> bool:
        """Start the AgentEduverse system"""
        if not self.check_environment():
            return False
        
        if components is None:
            components = ['mcp_server', 'master_agent', 'stock_agent', 'frontend']
        
        print("🤖 Starting AgentEduverse...")
        self.logger.info("Starting AgentEduverse system", extra_data={'components': components})
        
        # Start components in order
        startup_order = ['mcp_server', 'master_agent', 'stock_agent', 'frontend']
        components_to_start = [c for c in startup_order if c in components]
        
        for component in components_to_start:
            if not self.start_component(component):
                self.logger.error(f"Failed to start {component}")
                print(f"❌ Failed to start {component}")
                return False
            
            # Wait for component to be ready (except frontend)
            if component != 'frontend':
                print(f"⏳ Waiting for {self.components[component]['name']} to be ready...")
                if not asyncio.run(self.wait_for_component(component)):
                    self.logger.error(f"Component {component} failed health check")
                    print(f"❌ {self.components[component]['name']} failed health check")
                    return False
        
        print("✅ AgentEduverse started successfully!")
        self.logger.info("AgentEduverse system started successfully")
        
        # Print service URLs
        print("\n🌐 Service URLs:")
        for component in components_to_start:
            config = self.components[component]
            if component == 'frontend':
                print(f"   📊 Frontend: http://localhost:{config['port']}")
            else:
                print(f"   🔧 {config['name']}: http://localhost:{config['port']}")
        
        return True
    
    def stop_component(self, component_name: str):
        """Stop a specific component"""
        if component_name in self.processes:
            process_info = self.processes[component_name]
            process = process_info['process']
            config = process_info['config']
            
            self.logger.info(f"Stopping {config['name']}")
            print(f"🛑 Stopping {config['name']}...")
            
            try:
                process.terminate()
                process.wait(timeout=10)
                print(f"✅ {config['name']} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"⚠️  {config['name']} force killed")
            except Exception as e:
                self.logger.error(f"Error stopping {config['name']}: {str(e)}")
                print(f"❌ Error stopping {config['name']}: {str(e)}")
            
            del self.processes[component_name]
    
    def stop_system(self):
        """Stop all components"""
        print("🛑 Stopping AgentEduverse...")
        self.logger.info("Stopping AgentEduverse system")
        
        # Stop components in reverse order
        for component in reversed(list(self.processes.keys())):
            self.stop_component(component)
        
        print("✅ AgentEduverse stopped")
        self.logger.info("AgentEduverse system stopped")
    
    def monitor_system(self):
        """Monitor system health"""
        print("🔍 Monitoring system health...")
        
        while True:
            try:
                for component_name, process_info in self.processes.items():
                    process = process_info['process']
                    config = process_info['config']
                    
                    if process.poll() is not None:
                        print(f"⚠️  {config['name']} has stopped unexpectedly")
                        self.logger.warning(f"{config['name']} stopped unexpectedly")
                
                time.sleep(10)  # Check every 10 seconds
                
            except KeyboardInterrupt:
                print("\n🛑 Received interrupt signal")
                break

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AgentEduverse System Startup Script")
    parser.add_argument("--mcp-only", action="store_true", help="Start only MCP server")
    parser.add_argument("--agents-only", action="store_true", help="Start only agents (no frontend)")
    parser.add_argument("--frontend-only", action="store_true", help="Start only frontend")
    parser.add_argument("--monitor", action="store_true", help="Monitor system health after startup")
    
    args = parser.parse_args()
    
    # Determine which components to start
    if args.mcp_only:
        components = ['mcp_server']
    elif args.agents_only:
        components = ['mcp_server', 'master_agent', 'stock_agent']
    elif args.frontend_only:
        components = ['frontend']
    else:
        components = ['mcp_server', 'master_agent', 'stock_agent', 'frontend']
    
    # Create system manager
    manager = SystemManager()
    
    try:
        # Start system
        if manager.start_system(components):
            if args.monitor:
                manager.monitor_system()
            else:
                print("\n🎉 System is running! Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
        else:
            print("❌ Failed to start system")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        manager.stop_system()

if __name__ == "__main__":
    main()
