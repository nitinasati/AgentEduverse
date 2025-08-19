import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from utils.logger import logger

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class AgentAlert:
    agent_name: str
    alert_level: AlertLevel
    message: str
    timestamp: datetime
    details: Dict[str, Any]
    resolved: bool = False

class AgentMonitor:
    """Monitors agent behavior and ensures they stay within scope"""
    
    def __init__(self):
        self.logger = logger.get_logger("agent_monitor")
        self.alerts: List[AgentAlert] = []
        self.agent_stats: Dict[str, Dict[str, Any]] = {}
        self.monitoring_active = True
        
        # Monitoring thresholds
        self.thresholds = {
            'scope_violation_threshold': 0.3,  # Intent alignment confidence
            'rate_limit_threshold': 0.8,  # Percentage of rate limit
            'error_rate_threshold': 0.1,  # Error rate percentage
            'response_time_threshold': 30,  # Seconds
            'consecutive_failures_threshold': 3
        }
        
        # Agent endpoints to monitor
        self.agent_endpoints = {
            'master_agent': os.getenv('MASTER_AGENT_URL', 'http://localhost:8001'),
            'stock_agent': os.getenv('STOCK_AGENT_URL', 'http://localhost:8002')
        }
        
        self.logger.info("Agent Monitor initialized", extra_data={
            'thresholds': self.thresholds,
            'endpoints': self.agent_endpoints
        })
    
    async def start_monitoring(self):
        """Start continuous monitoring of all agents"""
        self.logger.info("Starting agent monitoring")
        
        while self.monitoring_active:
            try:
                await self._monitor_all_agents()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}", extra_data={
                    'error': str(e)
                })
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _monitor_all_agents(self):
        """Monitor all registered agents"""
        for agent_name, endpoint in self.agent_endpoints.items():
            try:
                await self._monitor_agent(agent_name, endpoint)
            except Exception as e:
                self.logger.error(f"Failed to monitor {agent_name}: {str(e)}", extra_data={
                    'agent_name': agent_name,
                    'error': str(e)
                })
    
    async def _monitor_agent(self, agent_name: str, endpoint: str):
        """Monitor a specific agent"""
        try:
            # Health check
            health_status = await self._check_agent_health(agent_name, endpoint)
            
            # Update agent stats
            if agent_name not in self.agent_stats:
                self.agent_stats[agent_name] = {
                    'health_checks': [],
                    'response_times': [],
                    'error_count': 0,
                    'last_seen': None,
                    'scope_violations': 0
                }
            
            stats = self.agent_stats[agent_name]
            stats['last_seen'] = datetime.now()
            
            # Check for issues
            await self._check_health_issues(agent_name, health_status)
            await self._check_rate_limits(agent_name, stats)
            await self._check_response_times(agent_name, stats)
            
        except Exception as e:
            self.logger.error(f"Agent monitoring failed for {agent_name}: {str(e)}", extra_data={
                'agent_name': agent_name,
                'error': str(e)
            })
    
    async def _check_agent_health(self, agent_name: str, endpoint: str) -> Dict[str, Any]:
        """Check agent health status"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = datetime.now()
                async with session.get(f"{endpoint}/health", timeout=10) as response:
                    response_time = (datetime.now() - start_time).total_seconds()
                    
                    if response.status == 200:
                        health_data = await response.json()
                        health_data['response_time'] = response_time
                        return health_data
                    else:
                        return {
                            'status': 'unhealthy',
                            'error': f"HTTP {response.status}",
                            'response_time': response_time
                        }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time': None
            }
    
    async def _check_health_issues(self, agent_name: str, health_status: Dict[str, Any]):
        """Check for health-related issues"""
        if health_status.get('status') != 'healthy':
            alert = AgentAlert(
                agent_name=agent_name,
                alert_level=AlertLevel.ERROR,
                message=f"Agent {agent_name} is unhealthy: {health_status.get('error', 'Unknown error')}",
                timestamp=datetime.now(),
                details=health_status
            )
            await self._create_alert(alert)
        
        # Check response time
        response_time = health_status.get('response_time')
        if response_time and response_time > self.thresholds['response_time_threshold']:
            alert = AgentAlert(
                agent_name=agent_name,
                alert_level=AlertLevel.WARNING,
                message=f"Agent {agent_name} response time is slow: {response_time:.2f}s",
                timestamp=datetime.now(),
                details={'response_time': response_time}
            )
            await self._create_alert(alert)
    
    async def _check_rate_limits(self, agent_name: str, stats: Dict[str, Any]):
        """Check if agent is approaching rate limits"""
        # This would need to be implemented based on your rate limiting logic
        pass
    
    async def _check_response_times(self, agent_name: str, stats: Dict[str, Any]):
        """Check response time trends"""
        response_times = stats.get('response_times', [])
        if len(response_times) > 10:
            avg_response_time = sum(response_times[-10:]) / 10
            if avg_response_time > self.thresholds['response_time_threshold']:
                alert = AgentAlert(
                    agent_name=agent_name,
                    alert_level=AlertLevel.WARNING,
                    message=f"Agent {agent_name} has consistently slow response times: {avg_response_time:.2f}s average",
                    timestamp=datetime.now(),
                    details={'avg_response_time': avg_response_time}
                )
                await self._create_alert(alert)
    
    async def _create_alert(self, alert: AgentAlert):
        """Create and log an alert"""
        self.alerts.append(alert)
        
        # Log the alert
        log_level = alert.alert_level.value.upper()
        self.logger.warning(f"AGENT ALERT [{log_level}]: {alert.message}", extra_data={
            'agent_name': alert.agent_name,
            'alert_level': alert.alert_level.value,
            'details': alert.details,
            'timestamp': alert.timestamp.isoformat()
        })
        
        # Send notification (implement based on your notification system)
        await self._send_notification(alert)
    
    async def _send_notification(self, alert: AgentAlert):
        """Send notification for critical alerts"""
        if alert.alert_level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
            # Implement notification logic (email, Slack, etc.)
            self.logger.critical(f"CRITICAL ALERT: {alert.message}", extra_data={
                'agent_name': alert.agent_name,
                'details': alert.details
            })
    
    def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get current status of an agent"""
        if agent_name not in self.agent_stats:
            return {'status': 'unknown', 'message': 'Agent not monitored'}
        
        stats = self.agent_stats[agent_name]
        return {
            'status': 'monitored',
            'last_seen': stats.get('last_seen'),
            'error_count': stats.get('error_count', 0),
            'scope_violations': stats.get('scope_violations', 0),
            'recent_alerts': [a for a in self.alerts if a.agent_name == agent_name and not a.resolved]
        }
    
    def get_all_alerts(self, resolved: bool = False) -> List[AgentAlert]:
        """Get all alerts, optionally filtered by resolved status"""
        return [alert for alert in self.alerts if alert.resolved == resolved]
    
    def resolve_alert(self, alert_index: int):
        """Mark an alert as resolved"""
        if 0 <= alert_index < len(self.alerts):
            self.alerts[alert_index].resolved = True
            self.logger.info(f"Alert resolved: {self.alerts[alert_index].message}")
    
    def stop_monitoring(self):
        """Stop the monitoring loop"""
        self.monitoring_active = False
        self.logger.info("Agent monitoring stopped")

# Global monitor instance
agent_monitor = AgentMonitor()
