"""Selector Agent for helping users choose preferred points of interest."""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq

from .base import BaseAgent, PointOfInterest, TravelPlanRequest

class SelectionRequest(BaseModel):
    """Request for selecting points of interest."""

class SelectorAgent(BaseAgent):
    """Agent responsible for helping users select points of interest."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b", temperature: float = 0.3):
        """Initialize the SelectorAgent with the specified Groq model."""
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.parser = JsonOutputParser()
        
        # Define the prompt template for auto-selection
        self.auto_select_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are a travel planning assistant helping to select the best points of interest
based on the user's preferences and constraints.

Given a list of points of interest and the user's travel request, select the most
appropriate ones that match their preferences, budget, and constraints.

For each selected POI, include:
- name: The name of the POI
- reason: Brief explanation of why it was selected
- priority: High, Medium, or Low priority

Return a JSON array of selected POIs with the above fields.
"""),
            ("human", """
Travel Request:
Destination: {destination}
Duration: {duration_days} days
Travel Style: {travel_style}
Budget: {budget}
Interests: {interests}
Constraints: {constraints}

Points of Interest:
{pois}

Please select the most suitable POIs for this trip, considering the user's preferences.
""")
        ])
        
        # Create the auto-select chain
        self.auto_select_chain = self.auto_select_prompt | self.llm | self.parser
    
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
    
    async def auto_select_pois(
        self, 
        travel_request: TravelPlanRequest,
        pois: List[PointOfInterest],
        max_selections: int = 10
    ) -> List[Dict[str, Any]]:
        """Automatically select POIs based on the travel request."""
        try:
            # Format the POIs for the prompt
            pois_str = "\n".join([
                f"- {poi.name} ({poi.category}): {poi.description} "
                f"[Duration: {poi.duration_minutes}min, Tags: {', '.join(poi.tags)}]"
                for poi in pois
            ])
            
            # Get the LLM response
            response = await self.auto_select_chain.ainvoke({
                "destination": travel_request.destination,
                "duration_days": travel_request.duration_days,
                "travel_style": ", ".join(travel_request.travel_style) if travel_request.travel_style else "Not specified",
                "budget": travel_request.budget or "Not specified",
                "interests": ", ".join(travel_request.interests) if hasattr(travel_request, 'interests') and travel_request.interests else "Not specified",
                "constraints": ", ".join(travel_request.constraints) if travel_request.constraints else "None",
                "pois": pois_str
            })
            
            # Map the response back to the original POIs
            selected_pois = []
            for item in response:
                # Find the original POI by name
                for poi in pois:
                    if poi.name.lower() == item["name"].lower():
                        selected_pois.append({
                            "poi": poi,
                            "reason": item.get("reason", "Selected based on preferences"),
                            "priority": item.get("priority", "Medium").capitalize()
                        })
                        break
            
            return selected_pois[:max_selections]
            
        except Exception as e:
            print(f"Error in auto-selecting POIs: {e}")
            # Fallback: return the first N POIs with default metadata
            return [{
                "poi": poi,
                "reason": "Selected by default",
                "priority": "Medium"
            } for poi in pois[:max_selections]]
    
    async def process(self, travel_request: TravelPlanRequest, pois: List[PointOfInterest]) -> List[Dict[str, Any]]:
        """Process the POI selection."""
        # For now, just use auto-selection
        # In a real app, you might want to add interactive selection here
        return await self.auto_select_pois(travel_request, pois)
    
    def _format_pois_for_prompt(self, pois: List[PointOfInterest]) -> str:
        """Format POIs as a string for the prompt."""
        result = []
        for i, poi in enumerate(pois, 1):
            poi_str = f"{i}. {poi.name} ({poi.category})\n"
            if poi.tags:
                poi_str += f"   Tags: {', '.join(poi.tags)}\n"
            if poi.description:
                poi_str += f"   {poi.description}\n"
            result.append(poi_str)
        
        return "\n".join(result)

# Example usage
if __name__ == "__main__":
    import asyncio
    from .base import TravelPlanRequest, PointOfInterest
    
    async def test_selector_agent():
        # Create test POIs
        test_pois = [
            PointOfInterest(
                name="Eiffel Tower",
                category="Landmark",
                duration_minutes=120,
                location="Champ de Mars, 5 Avenue Anatole France, 75007 Paris",
                tags=["iconic", "romantic", "view"],
                description="Iconic iron tower offering panoramic views of Paris."
            ),
            PointOfInterest(
                name="Louvre Museum",
                category="Museum",
                duration_minutes=180,
                location="Rue de Rivoli, 75001 Paris",
                tags=["art", "culture", "history"],
                description="World's largest art museum, home to the Mona Lisa."
            ),
            PointOfInterest(
                name="Montmartre",
                category="Neighborhood",
                duration_minutes=150,
                location="18th arrondissement, Paris",
                tags=["romantic", "artsy", "views"],
                description="Historic district with charming streets and the Sacré-Cœur Basilica."
            )
        ]
        
        # Create a test travel request
        travel_request = TravelPlanRequest(
            destination="Paris, France",
            duration_days=3,
            travel_style=["romantic", "cultural"],
            budget="mid-range",
            constraints=[]
        )
        
        # Initialize the agent
        agent = SelectorAgent()
        
        # Test auto-selection
        print("=== Testing Auto-Selection ===")
        auto_request = SelectionRequest(
            pois=test_pois,
            travel_request=travel_request,
            max_selections=2,
            auto_select=True
        )
        
        auto_result = await agent.process(auto_request)
        print("\nAuto-Selected POIs:")
        for poi in auto_result.selected_pois:
            print(f"- {poi.name}")
        print(f"\nReason: {auto_result.reason}")
        
        # Test interactive selection (uncomment to test)
        # print("\n=== Testing Interactive Selection ===")
        # interactive_request = SelectionRequest(
        #     pois=test_pois,
        #     travel_request=travel_request,
        #     max_selections=2,
        #     auto_select=False
        # )
        # interactive_result = await agent.process(interactive_request)
        # print("\nYou selected:")
        # for poi in interactive_result.selected_pois:
        #     print(f"- {poi.name}")
    
    asyncio.run(test_selector_agent())
