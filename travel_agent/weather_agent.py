"""WeatherAgent for providing weather and seasonal information for travel planning."""
from typing import Dict, List, Optional, Any, Tuple, Union
import os
import json
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta
import requests
import pytz
from geopy.geocoders import Nominatim

# Import the Tavily API key from environment variables
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY', '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')

@dataclass
class WeatherForecast:
    """Data class for weather forecast."""
    date: str
    temperature: Dict[str, float]  # min, max, feels_like, etc.
    condition: str  # e.g., 'sunny', 'rainy', 'cloudy'
    description: str  # Detailed description
    humidity: float  # Percentage
    wind_speed: float  # km/h
    precipitation: float  # mm
    icon: Optional[str] = None  # Weather icon code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'date': self.date,
            'temperature': self.temperature,
            'condition': self.condition,
            'description': self.description,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'precipitation': self.precipitation,
            'icon': self.icon
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WeatherForecast':
        """Create from dictionary."""
        return cls(**data)

@dataclass
class SeasonalInfo:
    """Data class for seasonal information."""
    location: str
    season: str
    avg_temperature: Dict[str, float]  # min, max
    conditions: List[str]  # Common weather conditions
    daylight_hours: float  # Average daylight hours
    recommendations: List[str]  # What to pack, activities, etc.
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'location': self.location,
            'season': self.season,
            'avg_temperature': self.avg_temperature,
            'conditions': self.conditions,
            'daylight_hours': self.daylight_hours,
            'recommendations': self.recommendations
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SeasonalInfo':
        """Create from dictionary."""
        return cls(**data)

class WeatherAgent:
    """Agent responsible for providing weather and seasonal information for travel planning."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.3, openweather_api_key: str = None):
        """
        Initialize the WeatherAgent.
        
        Args:
            model_name: Name of the LLM model (unused, for API compatibility)
            temperature: Temperature for model generation (unused, for API compatibility)
            openweather_api_key: Optional OpenWeatherMap API key
        """
        self.model_name = model_name
        self.temperature = temperature
        self.openweather_api_key = openweather_api_key or os.getenv('OPENWEATHER_API_KEY')
        self.geolocator = Nominatim(user_agent="travel_planner")
        self.cache_dir = Path("data/weather_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_file(self, location: str, is_forecast: bool = False) -> Path:
        """Get the cache file path for a location."""
        filename = f"{location.lower().replace(' ', '_')}_{'forecast' if is_forecast else 'seasonal'}.json"
        return self.cache_dir / filename
    
    def _load_from_cache(self, location: str, is_forecast: bool = False) -> Optional[Dict]:
        """Load data from cache if available and not expired."""
        cache_file = self._get_cache_file(location, is_forecast)
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                
            # Check if cache is still valid (24 hours for forecast, 7 days for seasonal)
            cache_time = datetime.fromisoformat(data['timestamp'])
            cache_expiry = 1 if is_forecast else 7  # days
            
            if (datetime.now() - cache_time).days < cache_expiry:
                return data['data']
                
        except Exception as e:
            print(f"Error loading from cache: {e}")
            
        return None
    
    def _save_to_cache(self, location: str, data: Any, is_forecast: bool = False) -> None:
        """Save data to cache."""
        try:
            cache_file = self._get_cache_file(location, is_forecast)
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'data': data if not hasattr(data, 'to_dict') else data.to_dict()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Error saving to cache: {e}")
    
    def get_coordinates(self, location: str) -> Optional[Tuple[float, float]]:
        """
        Get latitude and longitude for a location.
        
        Args:
            location: Name of the location
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        try:
            location = self.geolocator.geocode(location)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            print(f"Error getting coordinates: {e}")
        return None
    
    def get_season(self, date: datetime, hemisphere: str = 'northern') -> str:
        """
        Determine the season for a given date and hemisphere.
        
        Args:
            date: Date to check
            hemisphere: 'northern' or 'southern'
            
        Returns:
            Season name (spring, summer, fall, winter)
        """
        month = date.month
        day = date.day
        
        # For northern hemisphere
        if (month == 3 and day >= 20) or (month == 4) or (month == 5) or (month == 6 and day < 21):
            season = 'spring'
        elif (month == 6 and day >= 21) or (month == 7) or (month == 8) or (month == 9 and day < 23):
            season = 'summer'
        elif (month == 9 and day >= 23) or (month == 10) or (month == 11) or (month == 12 and day < 21):
            season = 'fall'
        else:
            season = 'winter'
        
        # Invert for southern hemisphere
        if hemisphere.lower() == 'southern':
            if season == 'summer':
                season = 'winter'
            elif season == 'winter':
                season = 'summer'
            elif season == 'spring':
                season = 'fall'
            elif season == 'fall':
                season = 'spring'
        
        return season
    
    def get_hemisphere(self, latitude: float) -> str:
        """
        Determine the hemisphere based on latitude.
        
        Args:
            latitude: Latitude coordinate
            
        Returns:
            'northern' or 'southern'
        """
        return 'northern' if latitude >= 0 else 'southern'
    
    async def get_weather_forecast(self, location: str, 
                                 start_date: datetime = None, 
                                 end_date: datetime = None) -> List[WeatherForecast]:
        """
        Get weather forecast for a location.
        
        Args:
            location: Name of the location
            start_date: Optional start date for forecast
            end_date: Optional end date for forecast (max 7-14 days depending on API)
            
        Returns:
            List of WeatherForecast objects
        """
        # Try to get from cache first
        cache_key = f"{location}_{start_date.date() if start_date else 'current'}_{end_date.date() if end_date else '7d'}"
        cached = self._load_from_cache(cache_key, is_forecast=True)
        if cached:
            return [WeatherForecast.from_dict(f) for f in cached]
        
        # Get coordinates for the location
        coords = self.get_coordinates(location)
        if not coords:
            print(f"Could not find coordinates for location: {location}")
            return []
            
        lat, lon = coords
        
        # Use OpenWeatherMap API if available
        if self.openweather_api_key:
            try:
                # Get current weather and forecast
                base_url = "https://api.openweathermap.org/data/2.5/onecall"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.openweather_api_key,
                    'units': 'metric',
                    'exclude': 'minutely,hourly,alerts'
                }
                
                response = requests.get(base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    forecasts = []
                    
                    # Process current weather
                    current = data.get('current', {})
                    if current:
                        forecast = WeatherForecast(
                            date=datetime.fromtimestamp(current['dt']).strftime('%Y-%m-%d'),
                            temperature={
                                'current': current['temp'],
                                'feels_like': current['feels_like'],
                                'min': current.get('temp_min', current['temp'] - 3),  # Estimate
                                'max': current.get('temp_max', current['temp'] + 3)   # Estimate
                            },
                            condition=current['weather'][0]['main'].lower(),
                            description=current['weather'][0]['description'],
                            humidity=current['humidity'],
                            wind_speed=current['wind_speed'] * 3.6,  # Convert m/s to km/h
                            precipitation=current.get('rain', {}).get('1h', 0) or current.get('snow', {}).get('1h', 0),
                            icon=current['weather'][0]['icon']
                        )
                        forecasts.append(forecast)
                    
                    # Process daily forecast (up to 7 days)
                    for day in data.get('daily', [])[:7]:  # Limit to 7 days
                        forecast = WeatherForecast(
                            date=datetime.fromtimestamp(day['dt']).strftime('%Y-%m-%d'),
                            temperature={
                                'min': day['temp']['min'],
                                'max': day['temp']['max'],
                                'morn': day['temp']['morn'],
                                'day': day['temp']['day'],
                                'eve': day['temp']['eve'],
                                'night': day['temp']['night']
                            },
                            condition=day['weather'][0]['main'].lower(),
                            description=day['weather'][0]['description'],
                            humidity=day['humidity'],
                            wind_speed=day['wind_speed'] * 3.6,  # Convert m/s to km/h
                            precipitation=day.get('rain', 0) or day.get('snow', 0),
                            icon=day['weather'][0]['icon']
                        )
                        forecasts.append(forecast)
                    
                    # Save to cache
                    self._save_to_cache(cache_key, [f.to_dict() for f in forecasts], is_forecast=True)
                    
                    return forecasts
                    
            except Exception as e:
                print(f"Error getting weather forecast from OpenWeatherMap: {e}")
        
        # Fallback to using Tavily API for weather information
        try:
            # Use Tavily API to search for weather information
            url = "https://api.tavily.com/search"
            query = f"{location} weather forecast {start_date.strftime('%B %Y') if start_date else ''}"
            
            response = requests.post(
                url,
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "include_answer": True,
                    "include_raw_content": False,
                    "max_results": 3
                },
                timeout=10
            )
            
            if response.status_code == 200:
                results = response.json()
                # In a real implementation, you would parse the search results
                # to extract weather forecast information
                # This is a simplified placeholder
                
                # For now, return a generic forecast
                forecast = WeatherForecast(
                    date=datetime.now().strftime('%Y-%m-%d'),
                    temperature={'min': 20, 'max': 30, 'feels_like': 25},
                    condition='sunny',
                    description='Mostly sunny with clear skies',
                    humidity=60,
                    wind_speed=10,
                    precipitation=0
                )
                
                return [forecast]
                
        except Exception as e:
            print(f"Error getting weather forecast from Tavily: {e}")
        
        return []
    
    async def get_seasonal_info(self, location: str, date: datetime = None) -> Optional[SeasonalInfo]:
        """
        Get seasonal information for a location.
        
        Args:
            location: Name of the location
            date: Optional date to check (defaults to current date)
            
        Returns:
            SeasonalInfo object or None if not found
        """
        if date is None:
            date = datetime.now()
        
        # Try to get from cache first
        cache_key = f"{location}_{date.month}"
        cached = self._load_from_cache(cache_key, is_forecast=False)
        if cached:
            return SeasonalInfo.from_dict(cached)
        
        # Get coordinates for the location
        coords = self.get_coordinates(location)
        if not coords:
            print(f"Could not find coordinates for location: {location}")
            return None
            
        lat, lon = coords
        hemisphere = self.get_hemisphere(lat)
        season = self.get_season(date, hemisphere)
        
        # Get timezone for the location
        try:
            # Use geopy to get timezone (this is a simplified approach)
            location_info = self.geolocator.reverse(f"{lat}, {lon}")
            timezone_str = location_info.raw.get('timezone', 'UTC')
            tz = pytz.timezone(timezone_str)
            
            # Calculate daylight hours (simplified)
            # In a real implementation, use a proper library like astral
            if season == 'summer':
                daylight_hours = 14  # Approximate for summer
            elif season == 'winter':
                daylight_hours = 10  # Approximate for winter
            else:
                daylight_hours = 12  # Approximate for spring/fall
            
            # Get seasonal temperature ranges (simplified)
            # In a real implementation, use historical weather data
            if season == 'summer':
                temp_range = {'min': 20, 'max': 35}  # Celsius
                conditions = ['sunny', 'warm', 'occasional thunderstorms']
                recommendations = [
                    'Pack light, breathable clothing',
                    'Bring sunscreen and a hat',
                    'Stay hydrated',
                    'Plan outdoor activities for early morning or late afternoon'
                ]
            elif season == 'winter':
                temp_range = {'min': -5, 'max': 10}  # Celsius
                conditions = ['cold', 'snowy', 'overcast']
                recommendations = [
                    'Pack warm clothing including a heavy coat',
                    'Bring waterproof boots',
                    'Check for winter weather advisories',
                    'Be prepared for possible travel delays'
                ]
            elif season == 'spring':
                temp_range = {'min': 10, 'max': 25}  # Celsius
                conditions = ['mild', 'rainy', 'changeable']
                recommendations = [
                    'Pack layers for changing temperatures',
                    'Bring a waterproof jacket',
                    'Be prepared for rain showers',
                    'Enjoy the spring blooms!'
                ]
            else:  # fall
                temp_range = {'min': 5, 'max': 20}  # Celsius
                conditions = ['cool', 'crisp', 'windy']
                recommendations = [
                    'Pack layers for cool mornings and evenings',
                    'Bring a warm jacket',
                    'Enjoy the fall foliage',
                    'Be prepared for rain'
                ]
            
            # Create the seasonal info object
            seasonal_info = SeasonalInfo(
                location=location,
                season=season,
                avg_temperature=temp_range,
                conditions=conditions,
                daylight_hours=daylight_hours,
                recommendations=recommendations
            )
            
            # Save to cache
            self._save_to_cache(cache_key, seasonal_info, is_forecast=False)
            
            return seasonal_info
            
        except Exception as e:
            print(f"Error getting seasonal info: {e}")
            return None
    
    async def get_travel_recommendations(self, location: str, 
                                       start_date: datetime = None,
                                       end_date: datetime = None) -> Dict[str, Any]:
        """
        Get comprehensive travel recommendations based on weather and season.
        
        Args:
            location: Name of the location
            start_date: Start date of the trip
            end_date: End date of the trip
            
        Returns:
            Dictionary with travel recommendations
        """
        if start_date is None:
            start_date = datetime.now()
        if end_date is None:
            end_date = start_date + timedelta(days=7)  # Default to 1 week
        
        # Get weather forecast and seasonal info
        forecast = await self.get_weather_forecast(location, start_date, end_date)
        seasonal_info = await self.get_seasonal_info(location, start_date)
        
        # Prepare recommendations
        recommendations = {
            'location': location,
            'trip_dates': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'duration_days': (end_date - start_date).days + 1
            },
            'weather_forecast': [f.to_dict() for f in forecast] if forecast else [],
            'seasonal_info': seasonal_info.to_dict() if seasonal_info else {},
            'packing_list': self._generate_packing_list(forecast, seasonal_info),
            'activity_suggestions': self._suggest_activities(forecast, seasonal_info)
        }
        
        return recommendations
    
    def _generate_packing_list(self, forecast: List[WeatherForecast], 
                             seasonal_info: Optional[SeasonalInfo]) -> List[str]:
        """Generate a packing list based on weather and season."""
        if not forecast and not seasonal_info:
            return ["Check the weather forecast closer to your travel date for specific packing recommendations."]
        
        packing_list = []
        
        # Start with seasonal recommendations if available
        if seasonal_info and seasonal_info.recommendations:
            packing_list.extend(seasonal_info.recommendations)
        
        # Add weather-specific items
        if forecast:
            # Check for rain
            if any(f.precipitation > 5 for f in forecast):  # More than 5mm of rain
                packing_list.append('Umbrella or rain jacket')
                packing_list.append('Waterproof shoes')
            
            # Check for cold temperatures
            min_temp = min(min(f.temperature.values()) for f in forecast)
            if min_temp < 10:  # Below 10°C
                packing_list.append('Warm coat')
                packing_list.append('Gloves and hat')
                packing_list.append('Thermal layers')
            
            # Check for hot temperatures
            max_temp = max(max(f.temperature.values()) for f in forecast)
            if max_temp > 30:  # Above 30°C
                packing_list.append('Sunscreen (SPF 30+)')
                packing_list.append('Sunglasses')
                packing_list.append('Light, breathable clothing')
                packing_list.append('Reusable water bottle')
        
        # Add general travel essentials
        essentials = [
            'Passport/ID',
            'Travel documents (tickets, reservations, etc.)',
            'Credit/debit cards and local currency',
            'Phone and charger',
            'Travel adapter (if international)',
            'Medications and first aid kit',
            'Toiletries'
        ]
        
        # Combine and deduplicate
        packing_list = list(dict.fromkeys(packing_list + essentials))
        
        return packing_list
    
    def _suggest_activities(self, forecast: List[WeatherForecast], 
                          seasonal_info: Optional[SeasonalInfo]) -> Dict[str, List[str]]:
        """Suggest activities based on weather and season."""
        suggestions = {
            'good_weather': [],
            'bad_weather': [],
            'seasonal': []
        }
        
        # Add seasonal activities if available
        if seasonal_info:
            if seasonal_info.season == 'summer':
                suggestions['seasonal'].extend([
                    'Beach outings',
                    'Hiking and outdoor adventures',
                    'Water sports',
                    'Outdoor festivals and concerts'
                ])
            elif seasonal_info.season == 'winter':
                suggestions['seasonal'].extend([
                    'Skiing or snowboarding',
                    'Winter hiking with proper gear',
                    'Hot springs',
                    'Museums and indoor attractions'
                ])
            elif seasonal_info.season == 'spring':
                suggestions['seasonal'].extend([
                    'Cherry blossom viewing',
                    'Garden tours',
                    'Outdoor photography',
                    'Bike tours'
                ])
            else:  # fall
                suggestions['seasonal'].extend([
                    'Fall foliage tours',
                    'Harvest festivals',
                    'Wine tasting',
                    'Hiking to see autumn colors'
                ])
        
        # Add weather-based activities if forecast is available
        if forecast:
            # Check for good weather (sunny/partly cloudy and not too hot/cold)
            good_weather_days = [
                f for f in forecast 
                if f.condition in ['clear', 'sunny', 'partly cloudy'] 
                and 15 <= max(f.temperature.values()) <= 28  # Comfortable temperature range
                and f.precipitation < 1  # No significant rain
            ]
            
            if good_weather_days:
                suggestions['good_weather'].extend([
                    'Outdoor sightseeing',
                    'Walking tours',
                    'Picnics in parks',
                    'Outdoor dining',
                    'Beach or pool days (if available)'
                ])
            
            # Check for bad weather (rain, storms, extreme temps)
            bad_weather_days = [
                f for f in forecast 
                if f.condition in ['rain', 'thunderstorm', 'snow', 'sleet']
                or f.precipitation >= 5  # Significant precipitation
                or max(f.temperature.values()) > 35  # Very hot
                or min(f.temperature.values()) < 0   # Freezing
            ]
            
            if bad_weather_days:
                suggestions['bad_weather'].extend([
                    'Museums and galleries',
                    'Indoor markets',
                    'Cooking classes',
                    'Spa or wellness centers',
                    'Shopping malls',
                    'Theater or cinema'
                ])
        
        # Remove duplicates
        for key in suggestions:
            suggestions[key] = list(dict.fromkeys(suggestions[key]))
        
        return suggestions
