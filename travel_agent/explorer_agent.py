"""Explorer Agent for discovering points of interest using Tavily API and LLM."""
from typing import List, Dict, Any, Optional, Union
import random
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from .base import BaseAgent, PointOfInterest, TravelPlanRequest, TravelStyle

class ExplorerAgent(BaseAgent):
    """Agent responsible for discovering points of interest using Tavily API and LLM."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b", temperature: float = 0.7):
        """Initialize the ExplorerAgent with Tavily API integration."""
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.parser = JsonOutputParser()
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.cache_dir = Path("data/cache/poi")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the prompt template for refining Tavily results
        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert travel guide. Your task is to process and refine points of interest 
from various sources into a consistent, structured format.

For each point of interest, extract and structure the following information:
- name: The name of the place/activity
- category: Type of place/activity (e.g., museum, park, restaurant)
- description: Brief description (1-2 sentences)
- duration_minutes: Estimated time needed in minutes
- location: Address or area
- tags: List of relevant tags (e.g., 'family-friendly', 'romantic', 'adventure')
- source: The source of this information
- rating: If available, the average rating (1-5)
- price_level: If available, the price level (1-4, with 4 being most expensive)
- opening_hours: If available, the typical opening hours

Consider the user's travel style, budget, and constraints when selecting and describing POIs.

Return a JSON array of objects with the above fields.
"""),
            ("human", """
Destination: {destination}
Travel Style: {travel_style}
Budget: {budget}
Interests: {interests}
Constraints: {constraints}

Please process the following search results into structured POI data:
{search_results}
""")
        ])
        
        # Create the chain
        self.chain = self.refine_prompt | self.llm | self.parser
    
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
    
    async def search_tavily(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for points of interest using Tavily API."""
        if not self.tavily_api_key:
            print("Warning: TAVILY_API_KEY not set. Using LLM fallback.")
            return []
            
        cache_key = f"tavily_{query.lower().replace(' ', '_')}_{max_results}.json"
        cache_file = self.cache_dir / cache_key
        
        # Check cache first
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
                # Check if cache is less than 7 days old
                cache_time = datetime.fromisoformat(cache_data.get('timestamp', '2000-01-01'))
                if datetime.now() - cache_time < timedelta(days=7):
                    return cache_data.get('results', [])
        
        # Call Tavily API
        url = "https://api.tavily.com/search"
        headers = {"Content-Type": "application/json"}
        data = {
            "api_key": self.tavily_api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answer": False,
            "include_raw_content": True,
            "max_results": max_results,
            "include_domains": ["tripadvisor.com", "lonelyplanet.com", "wikitravel.org"]
        }
        
        try:
            response = requests.post(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            results = response.json().get('results', [])
            
            # Cache the results
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'results': results
                }, f, indent=2)
                
            return results
            
        except Exception as e:
            print(f"Error calling Tavily API: {e}")
            return []
    
    def _generate_search_queries(self, travel_request: TravelPlanRequest) -> List[str]:
        """Generate search queries based on the travel request."""
        base_query = f"{travel_request.destination} "
        
        # Start with a base query for top attractions
        queries = [f"{base_query} top attractions"]
        
        # Add queries for each travel style
        if hasattr(travel_request, 'travel_style') and travel_request.travel_style:
            for style in travel_request.travel_style:
                queries.append(f"{base_query} best {style.value} activities")
        
        # Add queries for interests if any
        if hasattr(travel_request, 'interests') and travel_request.interests:
            for interest in travel_request.interests[:3]:  # Limit to top 3 interests
                queries.append(f"{base_query} best {interest} places")
                
        # Add queries for constraints if any
        if hasattr(travel_request, 'constraints') and travel_request.constraints:
            for constraint in travel_request.constraints[:2]:  # Limit to top 2 constraints
                queries.append(f"{base_query} {constraint} friendly places")
                
        return queries
    
    async def _fetch_from_apis(self, travel_request: TravelPlanRequest, num_pois: int) -> List[PointOfInterest]:
        """Fetch points of interest from external APIs."""
        if not self.tavily_api_key:
            return []
            
        # Generate search queries
        search_queries = self._generate_search_queries(travel_request)
        
        # Search for POIs using Tavily
        all_results = []
        for query in search_queries[:3]:  # Limit to top 3 queries
            results = await self.search_tavily(query, max_results=5)
            all_results.extend(results)
        
        if not all_results:
            return []
            
        # Refine results with LLM
        refined_pois = await self._refine_pois(all_results, travel_request, num_pois)
        
        # Convert to PointOfInterest objects
        pois = []
        for item in refined_pois:
            try:
                if isinstance(item, dict):
                    # Create PointOfInterest object with additional fields
                    poi = PointOfInterest(
                        name=item.get("name", ""),
                        category=item.get("category", "point_of_interest"),
                        duration_minutes=item.get("duration_minutes", 60),  # Default 1 hour
                        location=item.get("location", travel_request.destination),
                        tags=item.get("tags", []),
                        description=item.get("description", ""),
                        metadata={
                            "source": item.get("source", "tavily"),
                            "rating": item.get("rating"),
                            "price_level": item.get("price_level"),
                            "opening_hours": item.get("opening_hours")
                        }
                    )
                    pois.append(poi)
            except Exception as e:
                print(f"Error creating POI: {e}")
                continue
                
        return pois
    
    async def _refine_pois(self, search_results: List[Dict], travel_request: TravelPlanRequest, num_pois: int) -> List[Dict]:
        """Refine raw search results into structured POI data using LLM."""
        try:
            # Format the prompt with search results
            response = await self.chain.ainvoke({
                "destination": travel_request.destination,
                "travel_style": travel_request.travel_style if hasattr(travel_request, 'travel_style') else "not specified",
                "budget": travel_request.budget if hasattr(travel_request, 'budget') else "not specified",
                "interests": ", ".join(travel_request.interests) if hasattr(travel_request, 'interests') and travel_request.interests else "not specified",
                "constraints": ", ".join(travel_request.constraints) if hasattr(travel_request, 'constraints') and travel_request.constraints else "none",
                "search_results": json.dumps(search_results, indent=2)[:4000]  # Limit size
            })
            
            return response if isinstance(response, list) else []
            
        except Exception as e:
            print(f"Error refining POIs with LLM: {e}")
            return []
    
    async def _generate_with_llm(self, travel_request: TravelPlanRequest, num_pois: int) -> List[PointOfInterest]:
        """Generate POIs using LLM when API calls fail or are insufficient."""
        try:
            # Get travel styles as a string or use a default
            travel_styles = ", ".join([style.value for style in travel_request.travel_style]) if hasattr(travel_request, 'travel_style') and travel_request.travel_style else "Not specified"
            
            # Get constraints as a string or use a default
            constraints = ", ".join(travel_request.constraints) if hasattr(travel_request, 'constraints') and travel_request.constraints else "None"
            
            # Get budget or use a default
            budget = travel_request.budget if hasattr(travel_request, 'budget') else "Not specified"
            
            # Get interests as a string or use a default
            interests = ", ".join(travel_request.interests) if hasattr(travel_request, 'interests') and travel_request.interests else "Not specified"
            
            # Create a mock search result to guide the LLM
            mock_search_results = [
                {
                    "title": f"Top {num_pois} things to do in {travel_request.destination}",
                    "url": "https://example.com",
                    "content": f"Here are {num_pois} suggested activities in {travel_request.destination} based on your preferences."
                }
            ]
            
            # Use the same chain as _refine_with_llm but with our mock data
            response = await self.chain.ainvoke({
                "destination": travel_request.destination,
                "travel_style": travel_styles,
                "budget": budget,
                "interests": interests,
                "constraints": constraints,
                "search_results": json.dumps(mock_search_results, indent=2)
            })
            
            # Parse and validate the response
            if not isinstance(response, list):
                response = [response] if response else []
                
            # Convert to PointOfInterest objects
            pois = []
            for item in response:
                try:
                    if isinstance(item, dict):
                        # Ensure all required fields are present
                        if not all(field in item for field in ["name", "category", "duration_minutes"]):
                            continue
                            
                        # Create PointOfInterest object with metadata
                        poi = PointOfInterest(
                            name=item["name"],
                            category=item["category"],
                            duration_minutes=item["duration_minutes"],
                            location=item.get("location", travel_request.destination),
                            tags=item.get("tags", []),
                            description=item.get("description", ""),
                            metadata={
                                "source": item.get("source", "llm"),
                                "rating": item.get("rating"),
                                "price_level": item.get("price_level"),
                                "opening_hours": item.get("opening_hours")
                            }
                        )
                        pois.append(poi)
                except Exception as e:
                    print(f"Error parsing POI: {e}")
                    continue
                    
            return pois
            
        except Exception as e:
            print(f"Error generating POIs with LLM: {e}")
            return []
    
    async def process(self, travel_request: TravelPlanRequest) -> List[PointOfInterest]:
        """Process the travel request and return a list of points of interest.
        
        Args:
            travel_request: The travel plan request containing destination, duration, etc.
            
        Returns:
            A list of PointOfInterest objects matching the request.
        """
        # Calculate the target number of POIs (3-5 per day, but at least 5 total)
        num_days = max(1, travel_request.duration_days)
        target_pois = min(20, max(5, num_days * 3))
        
        # Try to fetch from APIs first
        pois = await self._fetch_from_apis(travel_request, target_pois)
        
        # If we didn't get enough POIs from APIs, generate more with LLM
        if len(pois) < target_pois:
            remaining = target_pois - len(pois)
            llm_pois = await self._generate_with_llm(travel_request, remaining)
            pois.extend(poi for poi in llm_pois if poi not in pois)
        
        # Ensure we have at least some POIs
        if not pois:
            print("Warning: No POIs found. Using fallback generation.")
            pois = await self._generate_with_llm(travel_request, 5)
            
        return pois

# Example usage
if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv
    
    async def main():
        # Load environment variables
        load_dotenv()
        
        # Create a test travel request
        test_request = TravelPlanRequest(
            destination="Paris, France",
            duration_days=5,
            travel_style="cultural",
            budget="medium",
            constraints=["wheelchair accessible"],
            interests=["museums", "history", "food"]
        )
        
        # Initialize the agent
        agent = ExplorerAgent()
        
        # Get POIs
        print("Searching for points of interest...")
        pois = await agent.process(test_request, num_pois=8)
        
        # Print results
        print(f"\nFound {len(pois)} POIs:")
        for i, poi in enumerate(pois, 1):
            print(f"\n{i}. {poi.name} ({poi.category})")
            print(f"   Duration: {poi.duration_minutes} minutes")
            print(f"   Location: {poi.location}")
            print(f"   Tags: {', '.join(poi.tags)}")
            print(f"   {poi.description}")
            
            # Print additional metadata if available
            if hasattr(poi, 'metadata') and poi.metadata:
                if 'rating' in poi.metadata and poi.metadata['rating']:
                    print(f"   Rating: {poi.metadata['rating']}/5")
                if 'price_level' in poi.metadata and poi.metadata['price_level']:
                    print(f"   Price Level: {'$' * int(poi.metadata['price_level'])}")
    
    # Run the example
    asyncio.run(main())
