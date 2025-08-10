"""Budget Calculation Agent for travel planning with currency conversion."""
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import date, datetime, timedelta
import random
import json
import os
from pathlib import Path
import requests

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from .models import (
    BudgetLevel, BudgetBreakdown, PointOfInterest, 
    TravelPlanRequest, TransportOption, TransportMode
)
from .base import BaseAgent

# Exchange rate API (free tier)
EXCHANGE_RATE_API = "https://api.exchangerate-api.com/v4/latest/INR"
EXCHANGE_RATE_CACHE_FILE = Path("data/exchange_rates.json")
EXCHANGE_RATE_CACHE_EXPIRY = 24 * 60 * 60  # 24 hours in seconds

class BudgetCalculationAgent(BaseAgent):
    """Agent responsible for calculating and managing travel budgets with currency support."""
    
    # Default daily budget estimates by budget level (per person) in INR
    # Updated to reflect more realistic prices in India for 2023-2024
    DEFAULT_DAILY_BUDGETS = {
        BudgetLevel.BUDGET: {
            "accommodation": 1500.0,    # Budget hotel/hostel
            "food": 1000.0,             # Street food and local restaurants
            "transport": 800.0,         # Local transport and short trips
            "activities": 1200.0,       # Entry fees to attractions
            "misc": 500.0               # Souvenirs, tips, etc.
        },
        BudgetLevel.MID_RANGE: {
            "accommodation": 4000.0,    # 3-star hotel or homestay
            "food": 2000.0,             # Mid-range restaurants
            "transport": 1500.0,        # Private cabs and transport
            "activities": 2500.0,       # Guided tours and activities
            "misc": 1000.0              # Souvenirs, tips, etc.
        },
        BudgetLevel.LUXURY: {
            "accommodation": 10000.0,   # 4-5 star hotel or luxury resort
            "food": 5000.0,             # Fine dining
            "transport": 4000.0,        # Private car with driver
            "activities": 6000.0,       # Premium experiences
            "misc": 3000.0              # Shopping, spa, etc.
        }
    }
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.2):
        """Initialize the BudgetCalculationAgent with the specified Groq model.
        
        Args:
            model_name: Name of the LLM model to use (default: llama-3.3-70b-versatile)
            temperature: Temperature for model generation (default: 0.2)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.parser = JsonOutputParser()
        self.default_currency = "INR"  # Set default currency to INR
        
        # Define the prompt template for budget estimation
        self.budget_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert travel budget planner. Your task is to help users estimate and manage 
their travel budget based on their destination, travel style, and preferences.

Consider the following when creating a budget:
1. Cost of living in the destination
2. Travel season (peak/off-peak)
3. User's budget level (budget/mid-range/luxury)
4. Travel style (backpacking, family, luxury, etc.)
5. Group size and any special requirements

Provide a detailed breakdown of estimated costs for:
- Accommodation
- Food & Drinks
- Transportation
- Activities & Attractions
- Miscellaneous expenses

Be realistic but flexible with the estimates.
"""),
            ("human", """
Estimate a travel budget with the following details:

Destination: {destination}
Duration: {duration_days} days
Budget Level: {budget_level}
Travel Style: {travel_style}
Group Size: {group_size}
Additional Notes: {additional_notes}

Please provide a detailed budget breakdown in the following JSON format:
{{
    "budget_breakdown": {{
        "accommodation": {{
            "daily_estimate": 100.0,
            "total_estimate": 500.0,
            "notes": "Budget hotel or hostel"
        }},
        "food": {{
            "daily_estimate": 50.0,
            "total_estimate": 250.0,
            "notes": "Meals at local restaurants"
        }},
        "transport": {{
            "daily_estimate": 30.0,
            "total_estimate": 150.0,
            "notes": "Public transport and taxis"
        }},
        "activities": {{
            "daily_estimate": 40.0,
            "total_estimate": 200.0,
            "notes": "Attractions and tours"
        }},
        "misc": {{
            "daily_estimate": 20.0,
            "total_estimate": 100.0,
            "notes": "Souvenirs and unexpected expenses"
        }}
    }},
    "total_estimated_cost": 1200.0,
    "budget_level": "mid-range",
    "currency": "USD",
    "recommendations": [
        "Consider booking accommodations with kitchen facilities to save on food costs",
        "Look for city passes that include public transport and attraction discounts"
    ]
}}
""")
        ])
        
        # Create the chain
        self.chain = self.budget_prompt | self.llm | self.parser
        
        # Initialize the budget chain after defining the prompt
        self.budget_chain = self.budget_prompt | self.llm | self.parser
        
        # Cache for storing budget estimates
        self.budget_cache = {}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the budget calculation request.
        
        Args:
            input_data: Dictionary containing budget calculation parameters
                - destination: Travel destination
                - duration_days: Number of travel days
                - budget_level: Budget level (e.g., 'budget', 'mid-range', 'luxury')
                - travel_style: List of travel styles
                - group_size: Number of travelers
                - additional_notes: Any additional notes or special requirements
                
        Returns:
            Dictionary containing the budget breakdown and recommendations
        """
        try:
            # Format the input for the LLM
            formatted_input = {
                "destination": input_data.get("destination", ""),
                "duration_days": input_data.get("duration_days", 1),
                "budget_level": input_data.get("budget_level", "mid-range"),
                "travel_style": ", ".join(input_data.get("travel_style", [])),
                "group_size": input_data.get("group_size", 1),
                "additional_notes": input_data.get("additional_notes", "")
            }
            
            # Get the LLM response
            response = await self.budget_chain.ainvoke(formatted_input)
            
            # Add metadata
            response["metadata"] = {
                "model": self.model_name,
                "timestamp": str(datetime.now().isoformat()),
                "currency": "INR"  # Default currency is INR
            }
            
            return {"status": "success", "data": response}
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to generate budget calculation: {str(e)}"
            }
    
    def _initialize_llm(self):
        """Initialize the appropriate LLM based on the model name."""
        from travel_agent.utils.model_config import ModelConfig
        
        # If model_name is None, get the default from ModelConfig
        if self.model_name is None:
            provider = ModelConfig.get_provider()
            self.model_name = ModelConfig.get_model_name(provider)
            
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
        
    def _get_exchange_rates(self) -> Dict[str, float]:
        """
        Get current exchange rates from API or cache.
        
        Returns:
            Dictionary of currency codes to exchange rates from INR
        """
        # Create data directory if it doesn't exist
        EXCHANGE_RATE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        # Try to load from cache first
        if EXCHANGE_RATE_CACHE_FILE.exists():
            try:
                with open(EXCHANGE_RATE_CACHE_FILE, 'r') as f:
                    cache_data = json.load(f)
                    
                # Check if cache is still valid
                last_updated = datetime.fromisoformat(cache_data.get('last_updated', '1970-01-01T00:00:00'))
                if (datetime.now() - last_updated).total_seconds() < EXCHANGE_RATE_CACHE_EXPIRY:
                    return cache_data.get('rates', {})
                    
            except Exception as e:
                print(f"Error loading exchange rate cache: {e}")
        
        # If cache is invalid or doesn't exist, fetch from API
        try:
            response = requests.get(EXCHANGE_RATE_API)
            if response.status_code == 200:
                data = response.json()
                rates = data.get('rates', {})
                
                # Save to cache
                cache_data = {
                    'last_updated': datetime.now().isoformat(),
                    'rates': rates
                }
                with open(EXCHANGE_RATE_CACHE_FILE, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                    
                return rates
                
        except Exception as e:
            print(f"Error fetching exchange rates: {e}")
            
        # Fallback to some default rates if API fails
        return {
            'USD': 0.012,  # 1 INR = 0.012 USD
            'EUR': 0.011,  # 1 INR = 0.011 EUR
            'GBP': 0.009,  # 1 INR = 0.009 GBP
            'JPY': 1.8,    # 1 INR = 1.8 JPY
            'AUD': 0.018,  # 1 INR = 0.018 AUD
            'CAD': 0.016,  # 1 INR = 0.016 CAD
            'INR': 1.0     # Base currency
        }
    
    def convert_currency(self, amount: float, from_currency: str, to_currency: str = 'INR') -> float:
        """
        Convert an amount from one currency to another.
        
        Args:
            amount: Amount to convert
            from_currency: Source currency code (e.g., 'USD', 'EUR')
            to_currency: Target currency code (default: 'INR')
            
        Returns:
            Converted amount in target currency
        """
        if from_currency.upper() == to_currency.upper():
            return amount
            
        rates = self._get_exchange_rates()
        
        # If either currency is not in the rates, return the original amount
        if from_currency.upper() not in rates or to_currency.upper() not in rates:
            print(f"Warning: Could not find exchange rate for {from_currency} or {to_currency}")
            return amount
            
        # Convert from source currency to INR, then to target currency
        inr_amount = amount / rates[from_currency.upper()]
        return inr_amount * rates[to_currency.upper()]
    
    def format_currency(self, amount: float, currency: str = 'INR') -> str:
        """
        Format a currency amount with the appropriate symbol and formatting.
        
        Args:
            amount: Amount to format
            currency: Currency code (e.g., 'USD', 'INR')
            
        Returns:
            Formatted currency string
        """
        currency_symbols = {
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'AUD': 'A$',
            'CAD': 'C$',
            'INR': '₹'
        }
        
        symbol = currency_symbols.get(currency.upper(), currency.upper())
        
        # Format with appropriate decimal places
        if currency.upper() in ['JPY', 'INR']:
            return f"{symbol}{amount:,.0f}"  # No decimal places for JPY and INR
        else:
            return f"{symbol}{amount:,.2f}"  # Two decimal places for others
    
    async def estimate_budget(
        self,
        destination: str,
        duration_days: int,
        budget_level: Union[BudgetLevel, str],
        travel_style: List[str],
        group_size: int = 1,
        additional_notes: str = "",
        target_currency: str = "INR"
    ) -> Dict[str, Any]:
        """
        Estimate a travel budget based on the given parameters.
        
        Args:
            destination: Travel destination
            duration_days: Number of days for the trip
            budget_level: Budget level (budget/mid-range/luxury)
            travel_style: List of travel styles (e.g., ["adventure", "family"])
            group_size: Number of people traveling
            additional_notes: Any additional notes or special requirements
            target_currency: Currency to use for the budget (default: 'INR')
            
        Returns:
            Dictionary containing the estimated budget breakdown
        """
        if travel_style is None:
            travel_style = ["leisure"]
            
        # Create a cache key
        cache_key = f"{destination}:{duration_days}:{budget_level.value}:{','.join(travel_style)}:{group_size}:{additional_notes}:{target_currency}"
        
        # Check cache first
        if cache_key in self.budget_cache:
            return self.budget_cache[cache_key]
            
        try:
            # Get the LLM response
            response = await self.chain.ainvoke({
                "destination": destination,
                "duration_days": duration_days,
                "budget_level": budget_level.value,
                "travel_style": ", ".join(travel_style),
                "group_size": group_size,
                "additional_notes": additional_notes
            })
            
            # Process and validate the response
            if isinstance(response, dict) and "budget_breakdown" in response:
                # Ensure all required fields are present
                breakdown = response["budget_breakdown"]
                required_categories = ["accommodation", "food", "transport", "activities", "misc"]
                
                for category in required_categories:
                    if category not in breakdown:
                        # Fall back to default values if any category is missing
                        return self._get_default_budget(destination, duration_days, budget_level, group_size)
                
                # Calculate total from breakdown
                total = sum(
                    item.get("total_estimate", 0) 
                    for item in breakdown.values()
                )
                
                # Ensure total is set
                response["total_estimated_cost"] = response.get("total_estimated_cost", total)
                
                # Convert currency if needed
                if target_currency.upper() != 'INR':
                    response = await self._convert_budget_currency(response, 'INR', target_currency)
                
                # Cache the result
                self.budget_cache[cache_key] = response
                
                return response
                
        except Exception as e:
            print(f"Error estimating budget: {e}")
            # Fall back to default budget calculation
            budget = await self._get_default_budget(destination, duration_days, budget_level, group_size, target_currency)
            
            # Convert currency if needed
            if target_currency.upper() != 'INR':
                budget = await self._convert_budget_currency(budget, 'INR', target_currency)
                
            return budget
    
    async def _convert_budget_currency(
        self,
        budget: Dict[str, Any],
        from_currency: str,
        to_currency: str
    ) -> Dict[str, Any]:
        """
        Convert all monetary values in a budget to a different currency.
        
        Args:
            budget: The budget dictionary to convert
            from_currency: Source currency code
            to_currency: Target currency code
            
        Returns:
            Budget with all monetary values converted to the target currency
        """
        if from_currency.upper() == to_currency.upper():
            return budget
            
        # Convert total estimated cost
        if 'total_estimated_cost' in budget:
            budget['total_estimated_cost'] = self.convert_currency(
                budget['total_estimated_cost'], from_currency, to_currency
            )
        
        # Convert all amounts in the budget breakdown
        if 'budget_breakdown' in budget:
            for category, details in budget['budget_breakdown'].items():
                if 'daily_estimate' in details:
                    details['daily_estimate'] = self.convert_currency(
                        details['daily_estimate'], from_currency, to_currency
                    )
                if 'total_estimate' in details:
                    details['total_estimate'] = self.convert_currency(
                        details['total_estimate'], from_currency, to_currency
                    )
                
                # Update notes to reflect currency conversion
                if 'notes' in details and 'converted from' not in details['notes'].lower():
                    details['notes'] = f"{details['notes']} (converted from {from_currency} to {to_currency})"
        
        # Update the currency in the budget
        budget['currency'] = to_currency.upper()
        
        return budget
        
    async def _get_default_budget(
        self,
        destination: str,
        duration_days: int,
        budget_level: BudgetLevel = BudgetLevel.MID_RANGE,
        group_size: int = 1,
        target_currency: str = 'INR'
    ) -> Dict[str, Any]:
        """
        Generate a default budget based on the budget level and duration.
        
        Args:
            destination: Travel destination (not used in default calculation)
            duration_days: Number of days for the trip
            budget_level: Budget level (budget/mid-range/luxury)
            group_size: Number of people traveling
            target_currency: Currency to use for the budget (default: 'INR')
            
        Returns:
            Dictionary containing the default budget breakdown
        """
        # Get the default daily budgets for the specified budget level (in INR)
        daily_budgets = self.DEFAULT_DAILY_BUDGETS.get(budget_level, self.DEFAULT_DAILY_BUDGETS[BudgetLevel.MID_RANGE])
        
        # Calculate total for each category
        # Convert to target currency if needed
        if target_currency.upper() != 'INR':
            budget = await self._convert_budget_currency(budget, 'INR', target_currency)
            
        return budget
    
    async def calculate_transport_costs(
        self,
        transport_plan: List[Dict[str, Any]],
        group_size: int = 1
    ) -> Dict[str, Any]:
        """
        Calculate transport costs based on the transport plan.
        
        Args:
            transport_plan: List of transport options between locations
            group_size: Number of people traveling
            
        Returns:
            Dictionary with transport cost details
        """
        if not transport_plan:
            return {
                "total_cost": 0.0,
                "breakdown": [],
                "recommendations": []
            }
        
        total_cost = 0.0
        breakdown = []
        
        for leg in transport_plan:
            if not leg.get("options"):
                continue
                
            # Get the recommended option or the first one available
            recommended_mode = leg.get("recommended_mode")
            option = next(
                (opt for opt in leg["options"] if opt.mode == recommended_mode),
                leg["options"][0]
            )
            
            # Calculate cost for the group
            leg_cost = option.cost * group_size
            total_cost += leg_cost
            
            breakdown.append({
                "from": leg.get("from", ""),
                "to": leg.get("to", ""),
                "mode": option.mode,
                "cost_per_person": option.cost,
                "total_cost": leg_cost,
                "duration_minutes": option.duration,
                "provider": option.provider or "N/A"
            })
        
        # Generate recommendations based on transport choices
        recommendations = [
            "Consider booking transport in advance for better rates.",
            "Check for group discounts or travel passes that might reduce costs."
        ]
        
        return {
            "total_cost": total_cost,
            "breakdown": breakdown,
            "recommendations": recommendations
        }
    
    async def calculate_accommodation_costs(
        self,
        destination: str,
        nights: int,
        budget_level: BudgetLevel = BudgetLevel.MID_RANGE,
        group_size: int = 1,
        room_type: str = "double"
    ) -> Dict[str, Any]:
        """
        Calculate accommodation costs.
        
        Args:
            destination: Travel destination
            nights: Number of nights
            budget_level: Budget level (budget/mid-range/luxury)
            group_size: Number of people
            room_type: Type of room (single, double, family, etc.)
            
        Returns:
            Dictionary with accommodation cost details
        """
        # In a real implementation, this would query a hotel API
        # For now, we'll use default values based on budget level
        
        # Base prices per night by budget level
        base_rates = {
            BudgetLevel.BUDGET: 30.0,
            BudgetLevel.MID_RANGE: 80.0,
            BudgetLevel.LUXURY: 200.0
        }
        
        # Room type multipliers
        room_multipliers = {
            "single": 1.0,
            "double": 1.5,
            "twin": 1.5,
            "family": 2.0,
            "suite": 2.5
        }
        
        # Get base rate and apply multipliers
        base_rate = base_rates.get(budget_level, 80.0)
        room_multiplier = room_multipliers.get(room_type.lower(), 1.5)
        
        # Calculate total cost
        price_per_night = base_rate * room_multiplier
        total_cost = price_per_night * nights
        
        # Adjust for group size (assuming 2 people per room)
        rooms_needed = max(1, (group_size + 1) // 2)
        total_cost *= rooms_needed
        
        return {
            "total_cost": total_cost,
            "price_per_night": price_per_night,
            "nights": nights,
            "rooms": rooms_needed,
            "room_type": room_type,
            "budget_level": budget_level.value,
            "recommendations": [
                "Consider booking accommodations with free cancellation for flexibility.",
                "Look for accommodations with included breakfast to save on food costs.",
                "Check for weekly or monthly rates for longer stays."
            ]
        }
