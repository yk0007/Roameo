"""MappingAgent for generating interactive maps and route directions."""
from typing import Dict, List, Optional, Any, Tuple
import os
import folium
import polyline
import requests
from pathlib import Path
from geopy.geocoders import Nominatim
from geopy.distance import geodesic

from .models import PointOfInterest, TransportOption

class MappingAgent:
    """Agent responsible for generating maps and route visualizations."""
    
    def __init__(self, mapbox_access_token: Optional[str] = None):
        """
        Initialize the MappingAgent.
        
        Args:
            mapbox_access_token: Optional Mapbox access token for enhanced map features
        """
        self.mapbox_access_token = mapbox_access_token or os.getenv('MAPBOX_ACCESS_TOKEN')
        self.geolocator = Nominatim(user_agent="travel_planner")
        
    def get_coordinates(self, location_name: str) -> Optional[Tuple[float, float]]:
        """
        Get latitude and longitude for a location name.
        
        Args:
            location_name: Name of the location
            
        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        try:
            location = self.geolocator.geocode(location_name)
            if location:
                return (location.latitude, location.longitude)
        except Exception as e:
            print(f"Error getting coordinates: {e}")
        return None
    
    def get_route_directions(self, origin: str, destination: str, 
                           mode: str = "driving") -> Optional[Dict]:
        """
        Get route directions between two points.
        
        Args:
            origin: Starting location
            destination: Destination location
            mode: Travel mode (driving, walking, cycling, transit)
            
        Returns:
            Dictionary with route information or None if failed
        """
        try:
            # Use OpenRouteService for routing (free tier available)
            headers = {
                'Accept': 'application/json, application/geo+json',
                'Content-Type': 'application/json',
            }
            
            # Get coordinates for origin and destination
            origin_coords = self.get_coordinates(origin)
            dest_coords = self.get_coordinates(destination)
            
            if not origin_coords or not dest_coords:
                return None
                
            # Prepare request data
            data = {
                "coordinates": [
                    [origin_coords[1], origin_coords[0]],  # Note: OpenRouteService uses [lon, lat]
                    [dest_coords[1], dest_coords[0]]
                ],
                "preference": "recommended",
                "instructions": True,
                "elevation": True
            }
            
            # Select appropriate profile based on mode
            profile = {
                "driving": "driving-car",
                "walking": "foot-walking",
                "cycling": "cycling-regular",
                "transit": "driving-car"  # OpenRouteService doesn't support transit directly
            }.get(mode.lower(), "driving-car")
            
            # Make the API request
            response = requests.post(
                f"https://api.openrouteservice.org/v2/directions/{profile}/geojson",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                return response.json()
                
        except Exception as e:
            print(f"Error getting route directions: {e}")
            
        return None
    
    def create_route_map(self, origin: str, destination: str, 
                        waypoints: Optional[List[str]] = None,
                        mode: str = "driving") -> Optional[folium.Map]:
        """
        Create an interactive map with a route from origin to destination.
        
        Args:
            origin: Starting location
            destination: Destination location
            waypoints: Optional list of waypoints
            mode: Travel mode (driving, walking, cycling, transit)
            
        Returns:
            folium.Map object or None if failed
        """
        try:
            # Get route directions
            route_data = self.get_route_directions(origin, destination, mode)
            if not route_data or 'features' not in route_data or not route_data['features']:
                return None
                
            # Extract coordinates from the route
            coordinates = route_data['features'][0]['geometry']['coordinates']
            # Convert [lon, lat] to (lat, lon) for folium
            route_coords = [(point[1], point[0]) for point in coordinates]
            
            # Create map centered on the first point
            map_center = route_coords[0]
            m = folium.Map(location=map_center, zoom_start=12)
            
            # Add the route to the map
            folium.PolyLine(
                route_coords,
                color='#1E90FF',
                weight=5,
                opacity=0.8
            ).add_to(m)
            
            # Add markers for origin, destination, and waypoints
            folium.Marker(
                route_coords[0],
                popup=f"<b>Origin:</b> {origin}",
                icon=folium.Icon(color='green', icon='play', prefix='fa')
            ).add_to(m)
            
            folium.Marker(
                route_coords[-1],
                popup=f"<b>Destination:</b> {destination}",
                icon=folium.Icon(color='red', icon='flag', prefix='fa')
            ).add_to(m)
            
            # Add waypoints if provided
            if waypoints:
                for i, waypoint in enumerate(waypoints, 1):
                    coords = self.get_coordinates(waypoint)
                    if coords:
                        folium.Marker(
                            coords,
                            popup=f"<b>Waypoint {i}:</b> {waypoint}",
                            icon=folium.Icon(color='orange', icon='map-marker')
                        ).add_to(m)
            
            # Add layer control
            folium.LayerControl().add_to(m)
            
            # Add fullscreen button
            folium.plugins.Fullscreen(
                position="topright",
                title="Expand me",
                title_cancel="Exit full screen",
                force_separate_button=True,
            ).add_to(m)
            
            return m
            
        except Exception as e:
            print(f"Error creating route map: {e}")
            return None
    
    def create_poi_map(self, pois: List[PointOfInterest], center: Optional[Tuple[float, float]] = None) -> Optional[folium.Map]:
        """
        Create a map with points of interest.
        
        Args:
            pois: List of PointOfInterest objects
            center: Optional center coordinates (lat, lon)
            
        Returns:
            folium.Map object or None if failed
        """
        try:
            if not pois:
                return None
                
            # If no center provided, use the first POI
            if not center and pois:
                center = (pois[0].latitude, pois[0].longitude)
            elif not center:
                center = (20.5937, 78.9629)  # Default to center of India
                
            # Create map
            m = folium.Map(location=center, zoom_start=12)
            
            # Add POI markers
            for poi in pois:
                if hasattr(poi, 'latitude') and hasattr(poi, 'longitude') and poi.latitude and poi.longitude:
                    popup_content = f"<b>{poi.name}</b>"
                    if hasattr(poi, 'description') and poi.description:
                        popup_content += f"<br/>{poi.description[:100]}..."
                    if hasattr(poi, 'category') and poi.category:
                        popup_content += f"<br/><i>Category: {poi.category}</i>"
                    
                    folium.Marker(
                        [poi.latitude, poi.longitude],
                        popup=popup_content,
                        tooltip=poi.name,
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(m)
            
            # Add layer control and fullscreen
            folium.LayerControl().add_to(m)
            folium.plugins.Fullscreen().add_to(m)
            
            return m
            
        except Exception as e:
            print(f"Error creating POI map: {e}")
            return None
    
    def save_map(self, map_obj: folium.Map, filename: str) -> str:
        """
        Save a folium map to an HTML file.
        
        Args:
            map_obj: folium.Map object to save
            filename: Output filename (without extension)
            
        Returns:
            Path to the saved file or empty string if failed
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("output/maps")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure filename has .html extension
            if not filename.endswith('.html'):
                filename += '.html'
                
            filepath = output_dir / filename
            map_obj.save(str(filepath))
            
            return str(filepath)
            
        except Exception as e:
            print(f"Error saving map: {e}")
            return ""
