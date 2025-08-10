"""TipsAgent for providing local tips and suggestions for destinations."""
from typing import Dict, List, Optional, Any, Tuple
import os
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import requests

from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Import the Tavily API key from environment variables
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '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')

@dataclass
class TravelTip:
    """Data class for travel tips."""
    category: str  # e.g., 'transportation', 'food', 'safety', 'culture', 'money', 'language', 'shopping'
    title: str
    description: str
    importance: str  # 'high', 'medium', 'low'
    applicable_seasons: List[str]  # 'spring', 'summer', 'fall', 'winter', 'all'
    tags: List[str]  # Additional tags for filtering
    source: str = "local_expert"  # Source of the tip
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'category': self.category,
            'title': self.title,
            'description': self.description,
            'importance': self.importance,
            'applicable_seasons': self.applicable_seasons,
            'tags': self.tags,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TravelTip':
        """Create from dictionary."""
        return cls(**data)

class TipsAgent:
    """Agent responsible for providing local tips and suggestions for destinations."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.7, use_llm: bool = True):
        """
        Initialize the TipsAgent.
        
        Args:
            model_name: Name of the LLM model to use
            temperature: Temperature for model generation (0.0 to 1.0)
            use_llm: Whether to use LLM for generating tips
        """
        self.model_name = model_name
        self.temperature = temperature
        self.use_llm = use_llm
        self.llm = None
        self.cache_file = Path("data/tips_cache.json")
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
        """Load tips cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except Exception as e:
            print(f"Error loading tips cache: {e}")
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save tips cache to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving tips cache: {e}")
    
    def _get_cache_key(self, location: str, category: str = None) -> str:
        """Generate a cache key for the given location and category."""
        key = location.lower().strip()
        if category:
            key += f":{category.lower().strip()}"
        return key
        self.use_llm = use_llm
        self.model_name = model_name
        self.llm = None
        self.cache_file = Path("data/travel_tips_cache.json")
        self.cache = {}
        
        if use_llm:
            self.llm = ChatOpenAI(
                model_name=model_name,
                temperature=0.7,
                openai_api_key=os.getenv('OPENAI_API_KEY')
            )
        
        # Load cache if exists
        self._load_cache()
    
    def _load_cache(self) -> None:
        """Load travel tips cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
        except Exception as e:
            print(f"Error loading tips cache: {e}")
            self.cache = {}
    
    def _save_cache(self) -> None:
        """Save travel tips cache to file."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Error saving tips cache: {e}")
    
    def _get_cache_key(self, location: str, categories: List[str] = None, 
                      season: str = None) -> str:
        """Generate a cache key for the given parameters."""
        key = location.lower().strip()
        if categories:
            key += ":" + ",".join(sorted([c.lower().strip() for c in categories]))
        if season:
            key += ":" + season.lower().strip()
        return key
    
    def _search_web_for_tips(self, location: str, categories: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search the web for travel tips about a location.
        
        Args:
            location: Name of the location
            categories: List of tip categories to search for
            
        Returns:
            List of search results
        """
        try:
            # Build the search query
            query_terms = [f"{location} travel tips"]
            
            if categories:
                query_terms.append(" OR ".join([f"{category} tips" for category in categories]))
            
            query = " ".join(query_terms)
            
            # Use Tavily API for web search
            url = "https://api.tavily.com/search"
            
            response = requests.post(
                url,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": True,
                    "include_raw_content": True,
                    "max_results": 10
                },
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                return results.get('results', [])
                
        except Exception as e:
            print(f"Error searching for travel tips: {e}")
            
        return []
    
    def _extract_tips_from_results(self, search_results: List[Dict[str, Any]]) -> List[TravelTip]:
        """
        Extract travel tips from search results.
        
        Args:
            search_results: List of search results
            
        Returns:
            List of TravelTip objects
        """
        # This is a simplified implementation - in a real app, you'd use more sophisticated parsing
        # or an LLM to extract structured information from the search results
        
        tips = []
        
        for result in search_results:
            title = result.get('title', '')
            content = result.get('content', '')
            url = result.get('url', '')
            
            # Simple heuristics to extract tips
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if not line or len(line) < 20:  # Skip very short lines
                    continue
                    
                # Try to extract a tip
                tip = self._parse_tip_line(line, source=url)
                if tip:
                    tips.append(tip)
                    
                # Limit the number of tips per result
                if len(tips) >= 5:
                    break
        
        return tips
    
    def _parse_tip_line(self, line: str, source: str = "") -> Optional[TravelTip]:
        """
        Parse a single line of text into a TravelTip if it contains a useful tip.
        
        Args:
            line: Line of text to parse
            source: Source URL of the tip
            
        Returns:
            TravelTip if a tip was found, None otherwise
        """
        # Skip lines that are too short or don't look like tips
        if len(line) < 20 or ' ' not in line.strip():
            return None
            
        # Simple keyword matching to categorize tips
        category = 'general'
        importance = 'medium'
        
        # Check for category keywords
        category_keywords = {
            'transport': ['bus', 'train', 'metro', 'subway', 'taxi', 'uber', 'lyft', 'rental', 'drive'],
            'food': ['eat', 'restaurant', 'cafe', 'food', 'dine', 'cuisine', 'drink', 'coffee', 'tea'],
            'safety': ['safe', 'danger', 'scam', 'pickpocket', 'police', 'emergency', 'avoid'],
            'culture': ['custom', 'dress', 'etiquette', 'religion', 'tradition', 'culture', 'local'],
            'money': ['price', 'cost', 'money', 'cash', 'credit card', 'ATM', 'tipping', 'tip'],
            'language': ['hello', 'thank you', 'please', 'language', 'speak', 'phrase'],
            'shopping': ['shop', 'market', 'bargain', 'haggle', 'souvenir', 'buy', 'purchase']
        }
        
        line_lower = line.lower()
        
        # Determine category
        for cat, keywords in category_keywords.items():
            if any(keyword in line_lower for keyword in keywords):
                category = cat
                break
        
        # Determine importance
        if any(word in line_lower for word in ['important', 'must', 'essential', 'critical', 'warning']):
            importance = 'high'
        elif any(word in line_lower for word in ['recommend', 'suggest', 'advise']):
            importance = 'medium'
        else:
            importance = 'low'
        
        # Create a title from the first few words
        words = line.split()
        title = ' '.join(words[:8]) + ('...' if len(words) > 8 else '')
        
        return TravelTip(
            category=category,
            title=title,
            description=line,
            importance=importance,
            applicable_seasons=['all'],
            tags=[category],
            source=source
        )
    
    async def get_tips_llm(self, location: str, categories: List[str] = None, 
                         season: str = None) -> List[TravelTip]:
        """
        Get travel tips using LLM.
        
        Args:
            location: Name of the location
            categories: List of tip categories to focus on
            season: Current season (optional)
            
        Returns:
            List of TravelTip objects
        """
        if not self.llm:
            return []
            
        try:
            # Prepare the prompt
            prompt = f"""You are a knowledgeable local travel expert. Provide a list of 10-15 useful travel tips 
            for someone visiting {location}. Organize the tips into categories like transportation, food, safety, 
            culture, money, language, and shopping. For each tip, include a brief title and description. 
            Also indicate the importance of each tip (high, medium, low) and which seasons it applies to."""
            
            if categories:
                prompt += f"\nFocus on these categories: {', '.join(categories)}."
                
            if season:
                prompt += f"\nThe current season is {season} - highlight seasonal tips if any."
            
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Please provide travel tips for {location} in a structured JSON format.")
            ]
            
            # Get response from LLM
            response = await self.llm.agenerate([messages])
            
            # Parse the response
            try:
                # Extract JSON from the response
                import re
                import json
                
                # Try to find JSON in the response
                json_match = re.search(r'```json\n(.*?)\n```', response.generations[0][0].text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If no code block, try to parse the whole response as JSON
                    json_str = response.generations[0][0].text
                
                tips_data = json.loads(json_str)
                
                return [
                    TravelTip(
                        category=item.get('category', 'general').lower(),
                        title=item.get('title', ''),
                        description=item.get('description', ''),
                        importance=item.get('importance', 'medium').lower(),
                        applicable_seasons=[s.lower() for s in item.get('applicable_seasons', ['all'])],
                        tags=item.get('tags', []),
                        source=item.get('source', 'local_expert')
                    )
                    for item in tips_data.get('tips', [])
                ]
                    
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                # Fallback to a simpler parsing approach
                return self._parse_llm_response_to_tips(response.generations[0][0].text)
                
        except Exception as e:
            print(f"Error getting tips from LLM: {e}")
            
        return []
    
    def _parse_llm_response_to_tips(self, text: str) -> List[TravelTip]:
        """
        Fallback method to parse LLM response when JSON parsing fails.
        
        Args:
            text: Raw text response from LLM
            
        Returns:
            List of TravelTip objects
        """
        tips = []
        current_category = 'general'
        
        # Split into lines and process each line
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check for category headers
            if line.lower().startswith('## '):
                category = line[3:].lower().strip()
                if 'transport' in category:
                    current_category = 'transportation'
                elif 'food' in category or 'dining' in category:
                    current_category = 'food'
                elif 'safety' in category or 'security' in category:
                    current_category = 'safety'
                elif 'culture' in category or 'etiquette' in category:
                    current_category = 'culture'
                elif 'money' in category or 'cost' in category or 'price' in category:
                    current_category = 'money'
                elif 'language' in category or 'phrase' in category:
                    current_category = 'language'
                elif 'shop' in category or 'market' in category:
                    current_category = 'shopping'
                else:
                    current_category = 'general'
                continue
                
            # Skip empty lines or lines that don't look like tips
            if not line or len(line) < 20 or ':' not in line:
                continue
                
            # Try to split into title and description
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
                
            title = parts[0].strip()
            description = parts[1].strip()
            
            # Determine importance
            importance = 'medium'
            if any(word in title.lower() for word in ['important', 'must', 'essential', 'critical', 'warning']):
                importance = 'high'
            elif any(word in title.lower() for word in ['recommend', 'suggest', 'advise']):
                importance = 'medium'
            else:
                importance = 'low'
            
            # Create and add the tip
            tips.append(TravelTip(
                category=current_category,
                title=title,
                description=description,
                importance=importance,
                applicable_seasons=['all'],
                tags=[current_category],
                source='local_expert'
            ))
            
            # Limit the number of tips
            if len(tips) >= 20:
                break
        
        return tips
    
    async def get_travel_tips(self, location: str, categories: List[str] = None, 
                            season: str = None, use_cache: bool = True) -> List[TravelTip]:
        """
        Get travel tips for a location.
        
        Args:
            location: Name of the location
            categories: List of tip categories to focus on
            season: Current season (optional)
            use_cache: Whether to use cached results if available
            
        Returns:
            List of TravelTip objects
        """
        # Check cache first
        cache_key = self._get_cache_key(location, categories, season)
        if use_cache and cache_key in self.cache:
            cached = self.cache[cache_key]
            # Check if cache is still valid (e.g., not older than 30 days)
            cache_time = datetime.fromisoformat(cached['timestamp'])
            if (datetime.now() - cache_time).days < 30:
                return [TravelTip.from_dict(item) for item in cached['tips']]
        
        # Get tips
        if self.use_llm and self.llm:
            tips = await self.get_tips_llm(location, categories, season)
        else:
            # Fallback to web search
            search_results = self._search_web_for_tips(location, categories)
            tips = self._extract_tips_from_results(search_results)
        
        # Update cache
        self.cache[cache_key] = {
            'timestamp': datetime.now().isoformat(),
            'location': location,
            'categories': categories,
            'season': season,
            'tips': [t.to_dict() for t in tips]
        }
        self._save_cache()
        
        return tips
    
    def get_tips_by_category(self, tips: List[TravelTip]) -> Dict[str, List[TravelTip]]:
        """
        Group tips by category.
        
        Args:
            tips: List of travel tips
            
        Returns:
            Dictionary mapping categories to lists of tips
        """
        categories = {}
        
        for tip in tips:
            category = tip.category.lower()
            if category not in categories:
                categories[category] = []
            categories[category].append(tip)
        
        return categories
    
    def filter_tips_by_season(self, tips: List[TravelTip], season: str) -> List[TravelTip]:
        """
        Filter tips by season.
        
        Args:
            tips: List of travel tips
            season: Current season (e.g., 'summer', 'winter')
            
        Returns:
            Filtered list of tips applicable to the given season
        """
        if not season:
            return tips
            
        season = season.lower()
        
        def is_applicable(tip: TravelTip) -> bool:
            if not tip.applicable_seasons:
                return True
                
            return any(s.lower() == 'all' or s.lower() == season 
                      for s in tip.applicable_seasons)
        
        return [t for t in tips if is_applicable(t)]
