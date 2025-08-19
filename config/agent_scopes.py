"""
Agent Scope Configuration

This file defines the scope boundaries, safety settings, and monitoring parameters
for each agent type in the AgentEduverse system.
"""

from typing import Dict, List, Any

# Master Agent Scope Configuration
MASTER_AGENT_SCOPE = {
    'allowed_domains': [
        'agent coordination',
        'task routing', 
        'system management',
        'agent discovery',
        'load balancing',
        'health monitoring',
        'workflow orchestration',
        'resource allocation'
    ],
    'forbidden_actions': [
        'execute financial transactions',
        'make investment decisions',
        'access sensitive data',
        'modify system files',
        'bypass security',
        'perform unauthorized actions',
        'access user credentials',
        'execute stock trades',
        'provide financial advice',
        'access personal data',
        'modify agent code'
    ],
    'max_iterations': 5,
    'timeout_seconds': 180,
    'rate_limits': {
        'requests_per_minute': 30,
        'requests_per_hour': 500
    },
    'safety_checks': {
        'input_validation': True,
        'output_filtering': True,
        'intent_verification': True,
        'scope_enforcement': True
    },
    'monitoring': {
        'health_check_interval': 30,
        'response_time_threshold': 15,
        'error_rate_threshold': 0.05,
        'scope_violation_threshold': 0.3
    }
}

# Stock Agent Scope Configuration
STOCK_AGENT_SCOPE = {
    'allowed_domains': [
        'stock analysis',
        'financial data',
        'investment research',
        'technical analysis',
        'market data',
        'company analysis',
        'financial statements',
        'market trends',
        'risk assessment',
        'portfolio analysis'
    ],
    'forbidden_actions': [
        'execute trades',
        'place orders',
        'access personal accounts',
        'modify system files',
        'bypass security',
        'access user credentials',
        'perform unauthorized actions',
        'make actual investments',
        'execute buy orders',
        'execute sell orders',
        'access trading accounts',
        'provide tax advice',
        'guarantee returns',
        'access personal financial data'
    ],
    'max_iterations': 8,
    'timeout_seconds': 240,
    'rate_limits': {
        'requests_per_minute': 20,
        'requests_per_hour': 300
    },
    'safety_checks': {
        'input_validation': True,
        'output_filtering': True,
        'intent_verification': True,
        'scope_enforcement': True
    },
    'monitoring': {
        'health_check_interval': 30,
        'response_time_threshold': 30,
        'error_rate_threshold': 0.1,
        'scope_violation_threshold': 0.3
    }
}

# General Agent Safety Patterns
SAFETY_PATTERNS = {
    'sensitive_data_patterns': [
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b\d{4}-\d{4}-\d{4}-\d{4}\b',  # Credit card
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{10,11}\b',  # Phone numbers
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b',  # IBAN
        r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP addresses
    ],
    'dangerous_commands': [
        'rm -rf',
        'format',
        'delete all',
        'drop database',
        'shutdown',
        'kill',
        'terminate',
        'wipe',
        'erase'
    ],
    'financial_risk_indicators': [
        'guaranteed return',
        'risk-free',
        'no risk',
        'sure thing',
        'can\'t lose',
        'guaranteed profit',
        '100% safe'
    ]
}

# Monitoring Configuration
MONITORING_CONFIG = {
    'global_thresholds': {
        'scope_violation_threshold': 0.3,
        'rate_limit_threshold': 0.8,
        'error_rate_threshold': 0.1,
        'response_time_threshold': 30,
        'consecutive_failures_threshold': 3
    },
    'alert_levels': {
        'info': {
            'scope_violation_threshold': 0.5,
            'response_time_threshold': 20
        },
        'warning': {
            'scope_violation_threshold': 0.3,
            'response_time_threshold': 15,
            'error_rate_threshold': 0.05
        },
        'error': {
            'scope_violation_threshold': 0.1,
            'response_time_threshold': 10,
            'error_rate_threshold': 0.1
        },
        'critical': {
            'scope_violation_threshold': 0.0,
            'response_time_threshold': 5,
            'error_rate_threshold': 0.2
        }
    },
    'notification_settings': {
        'email_alerts': True,
        'slack_alerts': False,
        'critical_only': False,
        'alert_cooldown_minutes': 15
    }
}

# Agent Type Mappings
AGENT_SCOPE_MAPPINGS = {
    'MasterAgent': MASTER_AGENT_SCOPE,
    'StockAgent': STOCK_AGENT_SCOPE,
    'default': {
        'allowed_domains': ['general assistance'],
        'forbidden_actions': [
            'execute trades',
            'access sensitive data',
            'modify system files',
            'bypass security'
        ],
        'max_iterations': 3,
        'timeout_seconds': 120,
        'rate_limits': {
            'requests_per_minute': 10,
            'requests_per_hour': 100
        },
        'safety_checks': {
            'input_validation': True,
            'output_filtering': True,
            'intent_verification': True,
            'scope_enforcement': True
        }
    }
}

def get_agent_scope(agent_name: str) -> Dict[str, Any]:
    """Get scope configuration for a specific agent"""
    return AGENT_SCOPE_MAPPINGS.get(agent_name, AGENT_SCOPE_MAPPINGS['default'])

def validate_scope_config(scope_config: Dict[str, Any]) -> bool:
    """Validate that a scope configuration is complete and valid"""
    required_keys = [
        'allowed_domains',
        'forbidden_actions', 
        'max_iterations',
        'timeout_seconds',
        'rate_limits',
        'safety_checks'
    ]
    
    for key in required_keys:
        if key not in scope_config:
            return False
    
    # Validate rate limits
    rate_limits = scope_config.get('rate_limits', {})
    if 'requests_per_minute' not in rate_limits or 'requests_per_hour' not in rate_limits:
        return False
    
    # Validate safety checks
    safety_checks = scope_config.get('safety_checks', {})
    required_safety_keys = ['input_validation', 'output_filtering', 'intent_verification', 'scope_enforcement']
    for key in required_safety_keys:
        if key not in safety_checks:
            return False
    
    return True
