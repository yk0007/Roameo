"""TravelSelections for managing user's travel choices and preferences."""
from typing import Dict, List, Optional, Any, Set
from datetime import date, datetime
import json
import os
from pathlib import Path

from .models import (
    PointOfInterest, TransportOption, BudgetLevel,
    TravelStyle, BudgetBreakdown, Activity, DailyItinerary, TravelItinerary
)

class TravelSelections:
    """Manages user's travel selections including POIs, restaurants, stays, and preferences."""
    
    def __init__(self, user_id: str = "default", data_dir: str = "data"):
        """
        Initialize the travel selections manager.
        
        Args:
            user_id: Unique identifier for the user
            data_dir: Directory to store user data
        """
        self.user_id = user_id
        self.data_dir = Path(data_dir)
        self.selections_file = self.data_dir / f"{user_id}_selections.json"
        
        # Initialize in-memory storage
        self.selected_pois: List[PointOfInterest] = []
        self.saved_restaurants: List[PointOfInterest] = []
        self.saved_accommodations: List[PointOfInterest] = []
        self.transport_options: List[TransportOption] = []
        self.itinerary: Optional[TravelItinerary] = None
        self.preferences: Dict[str, Any] = {
            "budget_level": BudgetLevel.MID_RANGE.value,
            "travel_styles": [TravelStyle.LEISURE.value],
            "dietary_restrictions": [],
            "accessibility_needs": [],
            "preferred_transport_modes": ["bus", "train"],
            "room_preferences": {},
            "preferred_currency": "INR"  # Default to INR
        }
        self.budget: Optional[BudgetBreakdown] = None
        self.trip_dates: Dict[str, date] = {
            "start_date": None,
            "end_date": None
        }
        
        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data if available
        self.load()
    
    def add_poi(self, poi: PointOfInterest, category: str = None) -> None:
        """
        Add a point of interest to the selections.
        
        Args:
            poi: Point of interest to add
            category: Optional category (e.g., 'restaurant', 'attraction')
        """
        # Check if POI already exists to avoid duplicates
        if not any(p.id == poi.id for p in self.selected_pois):
            self.selected_pois.append(poi)
            
            # Add to specific category if provided
            if category == 'restaurant' and not any(r.id == poi.id for r in self.saved_restaurants):
                self.saved_restaurants.append(poi)
            elif category == 'accommodation' and not any(a.id == poi.id for a in self.saved_accommodations):
                self.saved_accommodations.append(poi)
            
            self.save()
    
    def remove_poi(self, poi_id: str) -> bool:
        """
        Remove a point of interest from the selections.
        
        Args:
            poi_id: ID of the POI to remove
            
        Returns:
            True if POI was found and removed, False otherwise
        """
        removed = False
        
        # Remove from selected_pois
        self.selected_pois = [p for p in self.selected_pois if p.id != poi_id]
        
        # Remove from restaurants if present
        if any(r.id == poi_id for r in self.saved_restaurants):
            self.saved_restaurants = [r for r in self.saved_restaurants if r.id != poi_id]
            removed = True
            
        # Remove from accommodations if present
        if any(a.id == poi_id for a in self.saved_accommodations):
            self.saved_accommodations = [a for a in self.saved_accommodations if a.id != poi_id]
            removed = True
            
        if removed:
            self.save()
            
        return removed
    
    def add_transport_option(self, transport: TransportOption) -> None:
        """
        Add a transport option to the selections.
        
        Args:
            transport: Transport option to add
        """
        # Check if this exact transport option already exists
        if not any(self._transport_matches(t, transport) for t in self.transport_options):
            self.transport_options.append(transport)
            self.save()
    
    def _transport_matches(self, t1: TransportOption, t2: TransportOption) -> bool:
        """Check if two transport options are effectively the same."""
        return (
            t1.mode == t2.mode and
            t1.origin == t2.origin and
            t1.destination == t2.destination and
            t1.departure_time == t2.departure_time and
            t1.arrival_time == t2.arrival_time
        )
    
    def set_preferences(self, preferences: Dict[str, Any]) -> None:
        """
        Update user preferences.
        
        Args:
            preferences: Dictionary of preference updates
        """
        self.preferences.update(preferences)
        self.save()
    
    def set_budget(self, budget: BudgetBreakdown) -> None:
        """
        Set the budget for the trip.
        
        Args:
            budget: Budget breakdown
        """
        self.budget = budget
        self.save()
    
    def set_trip_dates(self, start_date: date, end_date: date) -> None:
        """
        Set the trip dates.
        
        Args:
            start_date: Trip start date
            end_date: Trip end date
        """
        self.trip_dates = {
            "start_date": start_date,
            "end_date": end_date
        }
        self.save()
    
    def set_itinerary(self, itinerary: TravelItinerary) -> None:
        """
        Set the travel itinerary.
        
        Args:
            itinerary: Travel itinerary
        """
        self.itinerary = itinerary
        self.save()
    
    def clear(self) -> None:
        """Clear all items from the selections."""
        self.selected_pois = []
        self.saved_restaurants = []
        self.saved_accommodations = []
        self.transport_options = []
        self.itinerary = None
        self.budget = None
        self.trip_dates = {"start_date": None, "end_date": None}
        self.save()
    
    def save(self) -> None:
        """Save the selections data to disk."""
        try:
            data = {
                "user_id": self.user_id,
                "selected_pois": [poi.dict() for poi in self.selected_pois],
                "saved_restaurants": [r.dict() for r in self.saved_restaurants],
                "saved_accommodations": [a.dict() for a in self.saved_accommodations],
                "transport_options": [t.dict() for t in self.transport_options],
                "preferences": self.preferences,
                "trip_dates": {
                    "start_date": self.trip_dates["start_date"].isoformat() if self.trip_dates["start_date"] else None,
                    "end_date": self.trip_dates["end_date"].isoformat() if self.trip_dates["end_date"] else None
                },
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Add budget if available
            if self.budget:
                data["budget"] = self.budget.dict()
                
            # Add itinerary if available
            if self.itinerary:
                data["itinerary"] = self.itinerary.dict()
            
            # Save to file
            with open(self.selections_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving selections data: {e}")
    
    def load(self) -> None:
        """Load selections data from disk."""
        if not self.selections_file.exists():
            return
            
        try:
            with open(self.selections_file, 'r') as f:
                data = json.load(f)
                
            # Load POIs
            self.selected_pois = [PointOfInterest(**poi) for poi in data.get("selected_pois", [])]
            self.saved_restaurants = [PointOfInterest(**r) for r in data.get("saved_restaurants", [])]
            self.saved_accommodations = [PointOfInterest(**a) for a in data.get("saved_accommodations", [])]
            
            # Load transport options
            self.transport_options = [TransportOption(**t) for t in data.get("transport_options", [])]
            
            # Load preferences
            self.preferences = data.get("preferences", self.preferences)
            
            # Load trip dates
            trip_dates = data.get("trip_dates", {})
            self.trip_dates = {
                "start_date": date.fromisoformat(trip_dates["start_date"]) if trip_dates.get("start_date") else None,
                "end_date": date.fromisoformat(trip_dates["end_date"]) if trip_dates.get("end_date") else None
            }
            
            # Load budget if available
            if "budget" in data:
                self.budget = BudgetBreakdown(**data["budget"])
                
            # Load itinerary if available
            if "itinerary" in data:
                self.itinerary = TravelItinerary(**data["itinerary"])
                
        except Exception as e:
            print(f"Error loading selections data: {e}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the travel selections.
        
        Returns:
            Dictionary with travel selections summary
        """
        return {
            "poi_count": len(self.selected_pois),
            "restaurant_count": len(self.saved_restaurants),
            "accommodation_count": len(self.saved_accommodations),
            "transport_options": len(self.transport_options),
            "has_itinerary": self.itinerary is not None,
            "has_budget": self.budget is not None,
            "trip_dates": self.trip_dates,
            "budget_level": self.preferences.get("budget_level", "not_set"),
            "preferred_currency": self.preferences.get("preferred_currency", "INR")
        }
