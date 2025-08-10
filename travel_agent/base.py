"""Base classes for travel agent system."""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar('T', bound=BaseModel)

class BaseAgent(ABC):
    """Base class for all agents in the travel planning system."""
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        """Process input and return output."""
        pass

class TravelStyle(str, Enum):
    """Enum for different travel styles."""
    ADVENTURE = "adventure"
    LUXURY = "luxury"
    BUDGET = "budget"
    FAMILY = "family"
    ROMANTIC = "romantic"
    SOLO = "solo"
    BUSINESS = "business"
    CULTURAL = "cultural"
    BEACH = "beach"
    NATURE = "nature"
    URBAN = "urban"
    ROAD_TRIP = "road_trip"
    BACKPACKING = "backpacking"
    FOODIE = "foodie"
    HISTORY = "history"
    ART = "art"
    SHOPPING = "shopping"
    WELLNESS = "wellness"

class TravelPlanRequest(BaseModel):
    """Structured travel plan request."""
    destination: str = Field(..., description="The travel destination")
    duration_days: int = Field(..., description="Number of travel days")
    travel_style: List[TravelStyle] = Field(
        default_factory=list,
        description="List of travel styles (e.g., 'romantic', 'adventure')"
    )
    budget: Optional[str] = Field(
        None,
        description="Budget level (e.g., 'budget', 'mid-range', 'luxury')"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description="Any constraints (e.g., 'family-friendly', 'wheelchair-accessible')"
    )

class Activity(BaseModel):
    """An activity in the travel itinerary."""
    name: str
    start_time: str
    end_time: str
    location: str
    description: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None

class PointOfInterest(BaseModel):
    """A point of interest for travel planning."""
    name: str
    category: str
    duration_minutes: int
    opening_hours: Optional[Dict[str, str]] = None
    location: str
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None

class DailyItinerary(BaseModel):
    """A single day's itinerary."""
    day: int
    activities: List[Dict[str, Any]] = Field(default_factory=list)

class TravelItinerary(BaseModel):
    """Complete travel itinerary."""
    destination: str
    duration_days: int
    daily_plans: List[DailyItinerary] = Field(default_factory=list)
    additional_notes: Optional[str] = None
