"""Transport Mode Agent for handling transportation planning."""
from typing import Dict, List, Optional, Any
from datetime import datetime, time, timedelta
import random

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

from .models import (
    TransportMode, TransportOption, PointOfInterest, 
    TravelPlanRequest, BudgetLevel
)
from .base import BaseAgent

class TransportModeAgent(BaseAgent):
    """Agent responsible for handling transportation planning and mode selection."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b", temperature: float = 0.3):
        """Initialize the TransportModeAgent."""
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.parser = JsonOutputParser()
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert transportation planner. Your task is to help users select the best 
transportation modes for their trip based on their preferences, budget, and itinerary.

Consider the following when making recommendations:
1. Distance and travel time between locations
2. User's budget level
3. User's travel style and preferences
4. Availability of transport options
5. Time of day and potential traffic conditions

For each leg of the journey, provide:
- Recommended transport mode
- Estimated duration
- Estimated cost
- Any additional notes or considerations
"""),
            ("human", """
Plan transportation for a trip with the following details:

Origin: {origin}
Destination: {destination}
Travel Date: {travel_date}
Budget Level: {budget_level}
Travel Style: {travel_style}
Preferred Transport Modes: {preferred_transport}
Additional Stops: {additional_stops}

Please provide a transportation plan that includes:
1. Recommended transport modes for each leg of the journey
2. Estimated travel times
3. Estimated costs
4. Any special considerations or tips

Format your response as a JSON object with the following structure:
{{
    "transport_plan": [
        {{
            "from": "origin",
            "to": "destination",
            "recommended_mode": "transport_mode",
            "options": [
                {{
                    "mode": "specific_mode",
                    "departure_time": "HH:MM",
                    "arrival_time": "HH:MM",
                    "duration_minutes": 120,
                    "cost": 50.0,
                    "provider": "company_name",
                    "notes": "any additional information"
                }}
            ]
        }}
    ],
    "total_estimated_cost": 100.0,
    "recommendations": "Brief explanation of the recommended plan"
}}
""")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm | self.parser
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the transportation planning request.
        
        Args:
            input_data: Dictionary containing transportation planning parameters
                - origin: Starting location
                - destination: Destination location
                - travel_date: Date of travel
                - budget_level: Budget level (e.g., 'budget', 'mid-range', 'luxury')
                - travel_style: List of travel styles
                - preferred_transport: List of preferred transport modes
                - additional_stops: List of additional stops
                
        Returns:
            Dictionary containing the transportation plan
        """
        try:
            # Format the input for the LLM
            formatted_input = {
                "origin": input_data.get("origin", ""),
                "destination": input_data.get("destination", ""),
                "travel_date": input_data.get("travel_date", ""),
                "budget_level": input_data.get("budget_level", "mid-range"),
                "travel_style": ", ".join(input_data.get("travel_style", [])),
                "preferred_transport": ", ".join(input_data.get("preferred_transport", [])),
                "additional_stops": ", ".join(input_data.get("additional_stops", []))
            }
            
            # Get the LLM response
            response = await self.chain.ainvoke(formatted_input)
            
            # Add metadata
            response["metadata"] = {
                "model": self.model_name,
                "timestamp": str(datetime.now().isoformat())
            }
            
            return {"status": "success", "data": response}
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to generate transportation plan: {str(e)}"
            }
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on the model name."""
        if "gemini" in self.model_name.lower():
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature
            )
        # Default to Groq for other models
        return ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature
        )
    
    async def plan_transport(
        self,
        origin: str,
        destination: str,
        travel_date: str,
        budget_level: str = "mid-range",
        travel_style: List[str] = None,
        preferred_transport: List[str] = None,
        additional_stops: List[str] = None,
        group_size: int = 1
    ) -> Dict[str, Any]:
        """
        Plan transportation for a trip.
        
        Args:
            origin: Starting location
            destination: Destination location
            travel_date: Date of travel in YYYY-MM-DD format
            budget_level: Budget level (budget, mid-range, luxury)
            travel_style: List of travel styles (e.g., ["adventure", "family"])
            preferred_transport: List of preferred transport modes
            additional_stops: List of additional stops to include
            group_size: Number of people traveling
            
        Returns:
            Dictionary containing the transportation plan
        """
        if travel_style is None:
            travel_style = ["leisure"]
        if preferred_transport is None:
            preferred_transport = ["bus", "train"]
        if additional_stops is None:
            additional_stops = []
            
        try:
            # Get the LLM response
            response = await self.chain.ainvoke({
                "origin": origin,
                "destination": destination,
                "travel_date": travel_date,
                "budget_level": budget_level,
                "travel_style": ", ".join(travel_style),
                "preferred_transport": ", ".join(preferred_transport),
                "additional_stops": ", ".join(additional_stops) if additional_stops else "None"
            })
            
            # Parse the response into TransportOption objects
            transport_plan = []
            total_cost = 0.0
            
            if isinstance(response, dict) and "transport_plan" in response:
                for leg in response["transport_plan"]:
                    options = []
                    for opt in leg.get("options", []):
                        transport_option = TransportOption(
                            mode=opt.get("mode", "bus"),
                            origin=leg.get("from", ""),
                            destination=leg.get("to", ""),
                            departure_time=opt.get("departure_time", "09:00"),
                            arrival_time=opt.get("arrival_time", "11:00"),
                            duration=opt.get("duration_minutes", 120),
                            cost=opt.get("cost", 0.0) * group_size,  # Adjust for group size
                            provider=opt.get("provider", ""),
                            notes=opt.get("notes", "")
                        )
                        options.append(transport_option)
                        total_cost += transport_option.cost
                    
                    transport_plan.append({
                        "from": leg.get("from", ""),
                        "to": leg.get("to", ""),
                        "recommended_mode": leg.get("recommended_mode", "bus"),
                        "options": options
                    })
            
            return {
                "transport_plan": transport_plan,
                "total_estimated_cost": total_cost,
                "recommendations": response.get("recommendations", "")
            }
            
        except Exception as e:
            print(f"Error planning transport: {e}")
            # Return a default transport plan in case of error
            return self._get_default_transport_plan(
                origin, destination, travel_date, budget_level, group_size
            )
    
    def _get_default_transport_plan(
        self,
        origin: str,
        destination: str,
        travel_date: str,
        budget_level: str,
        group_size: int
    ) -> Dict[str, Any]:
        """Generate a default transport plan in case of API failure."""
        # Simple fallback transport options
        modes = ["bus", "train", "flight"]
        base_costs = {"bus": 20, "train": 50, "flight": 200}
        durations = {"bus": 240, "train": 180, "flight": 60}  # in minutes
        
        # Adjust cost based on budget level
        budget_multiplier = {"budget": 0.7, "mid-range": 1.0, "luxury": 1.5}
        multiplier = budget_multiplier.get(budget_level.lower(), 1.0)
        
        options = []
        for mode in modes:
            cost = base_costs[mode] * multiplier * group_size
            duration = durations[mode]
            
            option = TransportOption(
                mode=mode,
                origin=origin,
                destination=destination,
                departure_time="09:00",
                arrival_time=self._add_minutes("09:00", duration),
                duration=duration,
                cost=cost,
                provider=f"Default {mode.capitalize()} Service",
                notes=f"Default {mode} option. Please verify times and availability."
            )
            options.append(option)
        
        return {
            "transport_plan": [{
                "from": origin,
                "to": destination,
                "recommended_mode": "train" if budget_level == "mid-range" else "bus",
                "options": options
            }],
            "total_estimated_cost": sum(opt.cost for opt in options) / len(options),  # Average cost
            "recommendations": "Default transport options. Please verify details with service providers."
        }
    
    def _add_minutes(self, time_str: str, minutes: int) -> str:
        """Add minutes to a time string and return as string."""
        try:
            if ':' in time_str:
                h, m = map(int, time_str.split(':'))
            else:
                h, m = int(time_str), 0
                
            total_minutes = h * 60 + m + minutes
            new_h = (total_minutes // 60) % 24
            new_m = total_minutes % 60
            return f"{new_h:02d}:{new_m:02d}"
        except:
            return "12:00"

    async def get_best_transport_option(
        self,
        origin: str,
        destination: str,
        preferred_modes: List[TransportMode] = None,
        max_duration: int = None,
        max_cost: float = None
    ) -> TransportOption:
        """
        Get the best transport option based on preferences and constraints.
        
        Args:
            origin: Starting location
            destination: Destination
            preferred_modes: List of preferred transport modes
            max_duration: Maximum allowed duration in minutes
            max_cost: Maximum allowed cost
            
        Returns:
            Best transport option based on the given constraints
        """
        if preferred_modes is None:
            preferred_modes = [TransportMode.BUS, TransportMode.TRAIN]
            
        # In a real implementation, this would query a transportation API
        # For now, we'll return a mock response
        options = await self.plan_transport(
            origin=origin,
            destination=destination,
            travel_date=date.today().isoformat(),
            preferred_transport=[m.value for m in preferred_modes]
        )
        
        if not options or not options.get("transport_plan"):
            return None
            
        # Get the first leg's options
        leg_options = options["transport_plan"][0].get("options", [])
        
        # Filter by max_duration if specified
        if max_duration is not None:
            leg_options = [opt for opt in leg_options if opt.duration <= max_duration]
            
        # Filter by max_cost if specified
        if max_cost is not None:
            leg_options = [opt for opt in leg_options if opt.cost <= max_cost]
            
        # Sort by duration (ascending) and cost (ascending)
        leg_options.sort(key=lambda x: (x.duration, x.cost))
        
        return leg_options[0] if leg_options else None
