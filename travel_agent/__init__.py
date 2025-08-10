"""
Agentic Travel Planner System

A modular, LLM-powered travel planning system using LangGraph and LangChain.
"""

# Load environment variables from .env file
from dotenv import load_dotenv
import os

# Load environment variables from .env file if it exists
load_dotenv()

# Verify required environment variables are set
required_vars = ['GOOGLE_API_KEY', 'GROQ_API_KEY']
for var in required_vars:
    if not os.getenv(var):
        print(f"Warning: Required environment variable {var} is not set. "
              f"Please set it in your environment or .env file.")

# Import key components to make them available at the package level
from .base import (
    BaseAgent,
    TravelPlanRequest,
    PointOfInterest,
    Activity,
    DailyItinerary,
    TravelItinerary,
    TravelStyle  # Make sure TravelStyle is imported
)

from .planner_agent import PlannerAgent
from .explorer_agent import ExplorerAgent
from .selector_agent import SelectorAgent
from .itinerary_agent import ItineraryAgent
from .formatter_agent import FormatterAgent
from .workflow import travel_planner_workflow, create_travel_planner_workflow

# Define what gets imported with 'from travel_agent import *'
__all__ = [
    'BaseAgent',
    'TravelPlanRequest',
    'PointOfInterest',
    'Activity',
    'DailyItinerary',
    'TravelItinerary',
    'PlannerAgent',
    'ExplorerAgent',
    'SelectorAgent',
    'ItineraryAgent',
    'FormatterAgent',
    'travel_planner_workflow',
    'create_travel_planner_workflow'
]

# Package metadata
__version__ = '0.1.0'
__author__ = 'Your Name'
__license__ = 'MIT'
__description__ = 'An AI-powered travel planning system using LLMs'
