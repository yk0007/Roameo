"""FoodAgent for suggesting must-try local foods and dining experiences."""
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from pathlib import Path
from dataclasses import dataclass
import requests
from datetime import datetime

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Import the Tavily API key from environment variables
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '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')

@dataclass
class FoodSuggestion:
    """Data class for food suggestions."""
    name: str
    description: str
    category: str  # e.g., 'breakfast', 'lunch', 'dinner', 'snack', 'dessert'
    price_range: str  # '$' to '$$$$' or 'budget'/'mid-range'/'luxury'
    dietary_info: List[str]  # e.g., ['vegetarian', 'vegan', 'gluten-free']
    best_time_to_try: str  # e.g., 'breakfast', 'lunch', 'dinner', 'anytime'
    best_season: List[str]  # e.g., ['summer', 'winter']
    must_try: bool = False  # If this is a must-try dish
    restaurant_suggestions: List[Dict[str, str]] = None  # List of dicts with name and location
    image_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'price_range': self.price_range,
            'dietary_info': self.dietary_info,
            'best_time_to_try': self.best_time_to_try,
            'best_season': self.best_season,
            'must_try': self.must_try,
            'restaurant_suggestions': self.restaurant_suggestions or [],
            'image_url': self.image_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FoodSuggestion':
        """Create from dictionary."""
        return cls(**data)

class FoodAgent:
    """Agent responsible for suggesting must-try local foods and dining experiences."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7, use_llm: bool = True):
        """
        Initialize the FoodAgent.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for model generation (0.0 to 1.0)
            use_llm: Whether to use LLM for generating suggestions
        """
        self.use_llm = use_llm
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.cache_file = Path("data/food_suggestions_cache.json")
        self.cache = {}
        
        if use_llm:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
        
        # Load cache if exists
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load food suggestions cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except Exception as e:
            print(f"Error loading food cache: {e}")
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save food suggestions cache to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving food cache: {e}")
    
    def _get_cache_key(self, location: str, season: str = None) -> str:
        """Generate a cache key for the given location and season."""
        key = location.lower().strip()
        if season:
            key += f":{season.lower().strip()}"
        return key
    
    def _search_web_for_foods(self, location: str) -> List[Dict[str, Any]]:
        """
        Search the web for popular foods in the given location.
        
        Args:
            location: Name of the location to search for
            
        Returns:
            List of food suggestions
        """
        try:
            # Use Tavily API for web search
            url = "https://api.tavily.com/search"
            query = f"must try local foods in {location} site:tripadvisor.com OR lonelyplanet.com OR timeout.com"
            
            response = requests.post(
                url,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": True,
                    "max_results": 5
                },
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                return results.get('results', [])
                
        except Exception as e:
            print(f"Error searching for foods: {e}")
            
        return []
    
    def _extract_food_info(self, search_results: List[Dict[str, Any]]) -> List[FoodSuggestion]:
        """
        Extract food information from search results.
        
        Args:
            search_results: List of search results from web search
            
        Returns:
            List of FoodSuggestion objects
        """
        # This is a simplified implementation - in a real app, you'd use more sophisticated parsing
        # or an LLM to extract structured information from the search results
        
        food_suggestions = []
        
        for result in search_results:
            # Extract basic information
            title = result.get('title', '')
            content = result.get('content', '')
            
            # Simple heuristic to identify food items
            # In a real app, you'd use more sophisticated NLP here
            if 'food' in title.lower() or 'dish' in title.lower() or 'cuisine' in title.lower():
                food_name = title.split('|')[0].split('-')[0].strip()
                
                # Create a basic food suggestion
                suggestion = FoodSuggestion(
                    name=food_name,
                    description=content[:200] + '...' if len(content) > 200 else content,
                    category='meal',  # Default category
                    price_range='$$',  # Default price range
                    dietary_info=[],
                    best_time_to_try='dinner',  # Default
                    best_season=['all'],
                    must_try=True,
                    restaurant_suggestions=[]
                )
                
                food_suggestions.append(suggestion)
        
        return food_suggestions
    
    async def get_food_suggestions_llm(self, location: str, season: str = None) -> List[FoodSuggestion]:
        """
        Get food suggestions using LLM.
        
        Args:
            location: Name of the location
            season: Current season (optional)
            
        Returns:
            List of FoodSuggestion objects
        """
        if not self.llm:
            return []
            
        try:
            # Prepare the prompt
            prompt = f"""You are a knowledgeable local food expert. Provide a list of 5-7 must-try local foods or 
            dishes for {location}. Include a brief description, category (breakfast, lunch, dinner, snack, dessert), 
            price range ($-$$$$), dietary information (vegetarian, vegan, gluten-free, etc.), best time to try 
            (breakfast, lunch, dinner, anytime), and best season to try (spring, summer, fall, winter, or all). 
            Also suggest 1-2 well-known places to try each dish in {location}."""
            
            if season:
                prompt += f" The current season is {season} - highlight seasonal specialties if any."
            
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Please provide food suggestions for {location} in a structured JSON format.")
            ]
            
            # Get response from LLM
            response = await self.llm.agenerate([messages])
            
            # Parse the response (this is simplified - in a real app, you'd need more robust parsing)
            try:
                # Extract JSON from the response
                import re
                json_str = re.search(r'```json\n(.*?)\n```', response.generations[0][0].text, re.DOTALL)
                if json_str:
                    food_data = json.loads(json_str.group(1))
                    return [
                        FoodSuggestion(
                            name=item.get('name', ''),
                            description=item.get('description', ''),
                            category=item.get('category', 'meal'),
                            price_range=item.get('price_range', '$$'),
                            dietary_info=item.get('dietary_info', []),
                            best_time_to_try=item.get('best_time_to_try', 'anytime'),
                            best_season=item.get('best_season', ['all']),
                            must_try=item.get('must_try', True),
                            restaurant_suggestions=item.get('restaurant_suggestions', []),
                            image_url=item.get('image_url')
                        )
                        for item in food_data.get('foods', [])
                    ]
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                
        except Exception as e:
            print(f"Error getting food suggestions from LLM: {e}")
            
        return []
    
    async def get_food_suggestions(self, location: str, season: str = None, 
                                 use_cache: bool = True) -> List[FoodSuggestion]:
        """
        Get food suggestions for a location.
        
        Args:
            location: Name of the location
            season: Current season (optional)
            use_cache: Whether to use cached results if available
            
        Returns:
            List of FoodSuggestion objects
        """
        # Check cache first
        cache_key = self._get_cache_key(location, season)
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            # Check if cache is still valid (e.g., not older than 30 days)
            cache_time = datetime.fromisoformat(cached['timestamp'])
            if (datetime.now() - cache_time).days < 30:
                return [FoodSuggestion.from_dict(item) for item in cached['suggestions']]
        
        # Get suggestions
        if self.use_llm and self.llm:
            suggestions = await self.get_food_suggestions_llm(location, season)
        else:
            # Fallback to web search
            search_results = self._search_web_for_foods(location)
            suggestions = self._extract_food_info(search_results)
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'season': season,
            'suggestions': [s.to_dict() for s in suggestions]
        }
        self._save_cache()
        
        return suggestions
    
    def get_food_categories(self, suggestions: List[FoodSuggestion]) -> Dict[str, List[FoodSuggestion]]:
        """
        Group food suggestions by category.
        
        Args:
            suggestions: List of food suggestions
            
        Returns:
            Dictionary mapping categories to lists of food suggestions
        """
        categories = {}
        
        for suggestion in suggestions:
            category = suggestion.category.lower()
            if category not in categories:
                categories[category] = []
            categories[category].append(suggestion)
        
        return categories
    
    def filter_by_dietary_restrictions(self, suggestions: List[FoodSuggestion], 
                                     restrictions: List[str]) -> List[FoodSuggestion]:
        """
        Filter food suggestions by dietary restrictions.
        
        Args:
            suggestions: List of food suggestions
            restrictions: List of dietary restrictions (e.g., ['vegetarian', 'gluten-free'])
            
        Returns:
            Filtered list of food suggestions
        """
        if not restrictions:
            return suggestions
            
        # Convert restrictions to lowercase for case-insensitive matching
        restrictions = [r.lower() for r in restrictions]
        
        def matches_restrictions(suggestion: FoodSuggestion) -> bool:
            # If no dietary info is available, include the suggestion
            if not suggestion.dietary_info:
                return True
                
            # Check if all restrictions are satisfied
            for restriction in restrictions:
                # Handle common variations
                if restriction == 'vegetarian':
                    if 'non-vegetarian' in suggestion.dietary_info:
                        return False
                    if 'vegetarian' not in suggestion.dietary_info and 'vegan' not in suggestion.dietary_info:
                        # If no info, be conservative and include it
                        pass
                elif restriction == 'vegan':
                    if 'vegan' not in suggestion.dietary_info:
                        return False
                elif restriction == 'gluten-free':
                    if 'gluten-free' not in suggestion.dietary_info and 'gluten' in suggestion.dietary_info:
                        return False
                # Add more restrictions as needed
                
            return True
        
        return [s for s in suggestions if matches_restrictions(s)]
