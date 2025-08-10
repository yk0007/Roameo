"""Data models for the travel planning system."""
from datetime import datetime, date, time
from enum import Enum
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator

class TravelStyle(str, Enum):
    ADVENTURE = "adventure"
    LEISURE = "leisure"
    BUSINESS = "business"
    FAMILY = "family"
    SOLO = "solo"
    COUPLE = "couple"
    BACKPACKING = "backpacking"
    LUXURY = "luxury"
    BUDGET = "budget"
    ROMANTIC = "romantic"
    CULTURAL = "cultural"
    BEACH = "beach"
    NATURE = "nature"
    URBAN = "urban"
    ROAD_TRIP = "road_trip"
    FOODIE = "foodie"
    HISTORY = "history"
    ART = "art"
    SHOPPING = "shopping"
    WELLNESS = "wellness"

class TransportMode(str, Enum):
    BIKE = "bike"
    CAR = "car"
    BUS = "bus"
    TRAIN = "train"
    FLIGHT = "flight"
    WALKING = "walking"
    TAXI = "taxi"

class BudgetLevel(str, Enum):
    BUDGET = "budget"
    MID_RANGE = "mid-range"
    LUXURY = "luxury"

class PointOfInterest(BaseModel):
    """Represents a point of interest."""
    id: str
    name: str
    category: str
    description: str
    location: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    duration_minutes: int = 60
    opening_hours: Optional[Dict[str, str]] = None
    price_range: Optional[str] = None
    rating: Optional[float] = None
    tags: List[str] = []
    image_url: Optional[str] = None

class Activity(BaseModel):
    """Represents an activity in the itinerary."""
    name: str
    start_time: str
    end_time: str
    location: str
    description: str
    category: Optional[str] = None
    notes: Optional[str] = None
    cost: Optional[float] = None
    transport_mode: Optional[TransportMode] = None
    transport_duration: Optional[int] = None  # in minutes
    transport_cost: Optional[float] = None

class DailyItinerary(BaseModel):
    """Represents a single day's itinerary."""
    day: int
    date: Optional[date] = None
    activities: List[Dict[str, Any]] = []
    total_cost: Optional[float] = None
    notes: Optional[str] = None

class TravelItinerary(BaseModel):
    """Represents a complete travel itinerary."""
    destination: str
    start_date: date
    end_date: date
    duration_days: int
    daily_plans: List[DailyItinerary] = []
    total_estimated_cost: float = 0.0
    budget_level: BudgetLevel = BudgetLevel.MID_RANGE
    travel_style: List[TravelStyle] = [TravelStyle.LEISURE]
    constraints: List[str] = []
    origin: Optional[str] = None
    transport_modes: List[TransportMode] = [TransportMode.BUS, TransportMode.TRAIN]

class TravelPlanRequest(BaseModel):
    """Represents a travel plan request."""
    destination: str
    duration_days: int
    travel_style: List[TravelStyle]
    budget: str
    interests: List[str] = []
    constraints: List[str] = []
    origin: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    preferred_transport: List[TransportMode] = [TransportMode.BUS, TransportMode.TRAIN]
    additional_stops: List[str] = []
    group_size: int = Field(
        default=1,
        description="Number of travelers in the group",
        ge=1,
        le=100
    )

class UserPreferences(BaseModel):
    """Stores user preferences and selections."""
    user_id: str
    saved_locations: List[PointOfInterest] = []
    saved_restaurants: List[PointOfInterest] = []
    saved_accommodations: List[PointOfInterest] = []
    travel_history: List[Dict[str, Any]] = []
    preferences: Dict[str, Any] = {}
    budget_preferences: Dict[str, float] = {
        "accommodation": 0.0,
        "food": 0.0,
        "transport": 0.0,
        "activities": 0.0,
        "total_budget": 0.0
    }

class BudgetBreakdown(BaseModel):
    """Detailed breakdown of travel costs."""
    accommodation: float = 0.0
    food: float = 0.0
    transport: float = 0.0
    activities: float = 0.0
    misc: float = 0.0
    total: float = 0.0

class TransportOption(BaseModel):
    """Represents a transport option between locations."""
    mode: TransportMode
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    duration: int  # in minutes
    cost: float
    provider: Optional[str] = None
    booking_reference: Optional[str] = None

class TravelPlanUpdate(BaseModel):
    """Represents an update to a travel plan."""
    plan_id: str
    updates: Dict[str, Any]
    reason: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
