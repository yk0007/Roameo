"""Main module for the Agentic Travel Planner system using LangGraph."""
import asyncio
import json
from typing import Optional, Dict, Any
from datetime import datetime

from .base import TravelItinerary
from .workflow import run_workflow

def print_itinerary(itinerary: TravelItinerary):
    """Print the travel itinerary in a user-friendly format."""
    if not itinerary or not hasattr(itinerary, 'daily_plans'):
        print("No valid itinerary to display.")
        return
        
    print(f"\n{'='*50}")
    print(f"YOUR {itinerary.duration_days}-DAY TRIP TO {itinerary.destination.upper()}")
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'-'*50}\n")
    
    for day_plan in itinerary.daily_plans:
        print(f"\n{'-'*20} DAY {day_plan.day} {'-'*20}")
        for activity in day_plan.activities:
            print(f"\n{activity['time']} - {activity['activity']}")
            if activity.get('notes'):
                print(f"   {activity['notes']}")
    
    if hasattr(itinerary, 'additional_notes') and itinerary.additional_notes:
        print(f"\n{'*'*50}")
        print("ADDITIONAL NOTES:")
        print(itinerary.additional_notes)

async def main():
    """Run the travel planner with an example request."""
    # Example user input
    user_input = """
    I want to visit Paris for 5 days. I'm interested in art, history, and good food.
    I have a medium budget and I'm traveling with my partner. We love walking and
    want to experience local culture. We don't like crowded tourist traps.
    """
    
    print("Welcome to the Agentic Travel Planner!")
    print("Planning your trip...\n")
    
    try:
        # Run the LangGraph workflow
        print("=== Starting Travel Planning Workflow ===")
        result = await run_workflow(user_input)
        
        if result["status"] == "error":
            print(f"\nError: {result['message']}")
            return
        
        # Print the itinerary
        print("\n=== Your Travel Itinerary ===")
        print_itinerary(result["itinerary"])
        
        # Optional: Save the itinerary to a file
        if result.get("itinerary"):
            with open("travel_itinerary.json", "w") as f:
                json.dump(result["itinerary"].dict(), f, indent=2)
            print("\n✓ Itinerary saved to 'travel_itinerary.json'")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please try again or provide more specific details about your trip.")

if __name__ == "__main__":
    asyncio.run(main())
