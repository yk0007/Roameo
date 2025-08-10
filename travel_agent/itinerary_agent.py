"""Itinerary Agent for creating day-by-day travel plans."""
from typing import List, Dict, Any, Optional
from datetime import time, datetime, timedelta, date
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .base import BaseAgent, PointOfInterest, TravelPlanRequest, DailyItinerary, TravelItinerary, Activity
from .utils import ModelConfig


class ItineraryAgent(BaseAgent):
    """Agent responsible for creating optimized daily itineraries."""
    
    def __init__(self, model_name: Optional[str] = None, temperature: float = 0.3):
        """
        Initialize the ItineraryAgent.
        
        Args:
            model_name: Optional model name. If not provided, uses the default from ModelConfig.
            temperature: Temperature for model generation (default: 0.3).
        """
        self.temperature = temperature
        self.model_config = ModelConfig.get_model_config()
        self.model_name = model_name or self.model_config['model_name']
        self.llm = ModelConfig.get_llm_instance(temperature=temperature)
        self.parser = JsonOutputParser()
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert travel planner creating engaging, realistic daily itineraries. Your task is to 
organize the selected points of interest into a logical, enjoyable schedule that considers travel times,
opening hours, and the natural flow of a day.

GUIDELINES:
1. Itinerary Structure:
   - Start with an engaging day introduction that sets the mood
   - Include realistic travel times between locations (15-60 mins depending on distance)
   - Schedule activities during appropriate hours (museums open 9AM-5PM, restaurants for meals, etc.)
   - Include meal breaks at reasonable times (breakfast 8-9AM, lunch 12:30-1:30PM, dinner 7-9PM)
   - Allow for 1-2 hours of free time each day
   - End each day by 10PM unless it's a nightlife activity

2. Activity Planning:
   - Group nearby activities to minimize travel time
   - Consider the best time to visit each location (e.g., early morning for popular sites)
   - Include 15-30 minute buffer between activities for travel and unexpected delays
   - Don't over-schedule - allow time to enjoy each location
   - Include unique local experiences and hidden gems

3. Travel Details:
   - Include specific travel instructions between locations
   - Note any entrance fees or reservation requirements
   - Suggest photo spots and best times for photography
   - Include local tips and cultural insights

4. Output Format:
   - Return a JSON object with 'days' as the top-level key
   - Each day should have 'day_number', 'date' (if available), and 'activities'
   - Each activity should include 'name', 'start_time', 'end_time', 'location', and 'description'
   - Include a 'travel_notes' field for any important travel information
   - Keep descriptions engaging but concise (1-2 sentences)
"""),
            ("human", """
Create a detailed {duration_days}-day itinerary for a trip to {destination} based on the following details:

TRAVEL DETAILS:
- Origin: {origin}
- Destination: {destination}
- Travel Dates: {start_date} to {end_date}
- Travel Style: {travel_style}
- Budget: {budget}
- Interests: {interests}
- Special Requirements: {constraints}

POINTS OF INTEREST:
{pois}

INSTRUCTIONS:
1. Create a realistic, well-paced daily schedule
2. Include travel time from origin to destination on Day 1
3. Include return travel on the last day
4. Group nearby activities to minimize travel time
5. Consider opening hours and best times to visit each location
6. Include meal breaks and free time
7. Provide engaging descriptions of each activity
8. Include local tips and recommendations

OUTPUT FORMAT:
{{
  "days": [
    {{
      "day_number": 1,
      "date": "YYYY-MM-DD",
      "activities": [
        {{
          "name": "Activity Name",
          "start_time": "HH:MM AM/PM",
          "end_time": "HH:MM AM/PM",
          "location": "Location Name",
          "description": "Engaging 1-2 sentence description of the activity and what makes it special."
        }}
      ],
      "travel_notes": "Any important travel information for the day"
    }}
  ]
}}

IMPORTANT: Be realistic about what can be accomplished in a day. Don't over-schedule!
""")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm | self.parser
    
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
    
    async def _generate_with_llm(
        self, 
        travel_request: TravelPlanRequest, 
        selected_pois: List[Dict[str, Any]]
    ) -> List[DailyItinerary]:
        """Generate an itinerary using the LLM with enhanced prompt and validation."""
        try:
            # Debug: Print the type and content of selected_pois
            print(f"DEBUG: Type of selected_pois: {type(selected_pois)}")
            if selected_pois:
                print(f"DEBUG: First POI type: {type(selected_pois[0])}")
                if hasattr(selected_pois[0], 'name'):
                    print(f"DEBUG: First POI name: {selected_pois[0].name}")
                elif isinstance(selected_pois[0], dict) and 'poi' in selected_pois[0]:
                    print(f"DEBUG: First POI (from dict): {selected_pois[0]['poi'].name}")
            
            # Format the POIs for the prompt with more detailed information
            pois_list = []
            for item in selected_pois:
                try:
                    if isinstance(item, dict) and 'poi' in item:
                        poi = item['poi']
                        poi_dict = poi.dict() if hasattr(poi, 'dict') else poi
                        name = poi_dict.get('name', 'Unknown')
                        category = poi_dict.get('category', 'Sightseeing')
                        description = poi_dict.get('description', 'No description available')
                        duration = poi_dict.get('duration_minutes', 60)
                        location = poi_dict.get('location', travel_request.destination)
                        
                        # Add more detailed POI information
                        poi_str = f"- {name} ({category})\n"
                        poi_str += f"  • Location: {location}\n"
                        poi_str += f"  • Duration: {duration} minutes\n"
                        
                        # Add optional fields if available
                        if 'opening_hours' in poi_dict and poi_dict['opening_hours']:
                            poi_str += f"  • Hours: {poi_dict['opening_hours']}\n"
                        if 'cost' in poi_dict and poi_dict['cost']:
                            poi_str += f"  • Cost: ₹{poi_dict['cost']}\n"
                        if 'rating' in poi_dict and poi_dict['rating']:
                            poi_str += f"  • Rating: {poi_dict['rating']}/5\n"
                            
                        poi_str += f"  • Description: {description}\n"
                        
                        # Add priority if available
                        if 'priority' in item:
                            poi_str += f"  • Priority: {item['priority']}\n"
                            
                        pois_list.append(poi_str)
                        
                except Exception as e:
                    print(f"Error formatting POI: {e}")
                    continue
            
            pois_str = "\n".join(pois_list)
            
            # If no POIs were found, use the fallback method
            if not pois_str.strip():
                print("No POIs found, using fallback itinerary")
                return await self._create_basic_itinerary(travel_request, selected_pois)
            
            # Calculate trip dates if available
            start_date = travel_request.start_date.strftime("%Y-%m-%d") if hasattr(travel_request, 'start_date') and travel_request.start_date else "Not specified"
            end_date = ""
            if hasattr(travel_request, 'start_date') and travel_request.start_date and hasattr(travel_request, 'duration_days'):
                end_date = (travel_request.start_date + timedelta(days=travel_request.duration_days - 1)).strftime("%Y-%m-%d")
            
            # Get origin or use a default
            origin = getattr(travel_request, 'origin', 'Your starting location')
            
            # Prepare the input for the chain with all required fields
            chain_input = {
                "origin": origin,
                "destination": travel_request.destination,
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": travel_request.duration_days,
                "travel_style": ", ".join(travel_request.travel_style) if hasattr(travel_request, 'travel_style') and travel_request.travel_style else "Not specified",
                "budget": travel_request.budget if hasattr(travel_request, 'budget') and travel_request.budget else "Not specified",
                "interests": ", ".join(travel_request.interests) if hasattr(travel_request, 'interests') and travel_request.interests else "Not specified",
                "constraints": ", ".join(travel_request.constraints) if hasattr(travel_request, 'constraints') and travel_request.constraints else "None",
                "pois": pois_str
            }
            
            print("DEBUG: Sending enhanced prompt to LLM...")
            
            print(f"DEBUG: Chain input: {chain_input}")
            
            # Get the LLM response
            response = await self.chain.ainvoke(chain_input)
            
            # Debug: Print raw LLM response
            print(f"DEBUG: Raw LLM response type: {type(response)}")
            if hasattr(response, 'content'):
                print(f"DEBUG: LLM response content type: {type(response.content)}")
                print(f"DEBUG: LLM response content (first 500 chars): {str(response.content)[:500]}")
            
            # Parse the response into a structured format
            try:
                print("DEBUG: Attempting to parse LLM response with parser...")
                parsed_response = self.parser.parse(response.content) if hasattr(response, 'content') else response
                print(f"DEBUG: Successfully parsed LLM response. Type: {type(parsed_response)}")
                
                # Log detailed information about the parsed response
                if isinstance(parsed_response, dict):
                    print(f"DEBUG: Parsed response keys: {list(parsed_response.keys())}")
                    # Log first few items of lists if they exist
                    for key, value in parsed_response.items():
                        if isinstance(value, (list, tuple)) and value:
                            print(f"DEBUG:   {key} (first 3 items): {value[:3] if len(value) > 3 else value}")
                        elif isinstance(value, dict):
                            print(f"DEBUG:   {key} (dict keys): {list(value.keys())}")
                        else:
                            print(f"DEBUG:   {key}: {value}")
                
                # Parse the response into DailyItinerary objects
                print("DEBUG: Calling _parse_itinerary_response with parsed response...")
                result = self._parse_itinerary_response(parsed_response, travel_request)
                print(f"DEBUG: _parse_itinerary_response returned: {type(result)}")
                if hasattr(result, 'daily_plans'):
                    print(f"DEBUG: Itinerary contains {len(result.daily_plans)} daily plans")
                return result
                
            except Exception as e:
                print(f"ERROR: Failed to parse LLM response: {e}")
                import traceback
                traceback.print_exc()
                
                # Log more details about the problematic response
                response_content = response.content if hasattr(response, 'content') else response
                print(f"DEBUG: Response type: {type(response_content)}")
                print(f"DEBUG: Response content (first 1000 chars): {str(response_content)[:1000]}")
                
                # Fall back to basic itinerary if parsing fails
                print("Falling back to basic itinerary due to parsing error")
                return self._create_basic_itinerary(travel_request, selected_pois)
            
        except Exception as e:
            print(f"Error generating itinerary with LLM: {e}")
            # Fallback to a basic itinerary
            return self._create_basic_itinerary(travel_request, selected_pois)
    
    def _calculate_optimal_start_time(self, travel_request: TravelPlanRequest, day_num: int) -> time:
        """Calculate optimal start time based on trip context and day number."""
        try:
            # First day: start later if it's a travel day
            if day_num == 1 and travel_request.origin and travel_request.origin != travel_request.destination:
                # Assume 2 hours for check-in and travel to first activity
                return time(11, 0)  # Start at 11 AM on first day
            
            # Last day: start earlier to accommodate return travel
            if day_num == travel_request.duration_days and travel_request.origin != travel_request.destination:
                return time(8, 0)  # Start at 8 AM on last day
                
            # Middle days: use a reasonable default
            return time(9, 0)  # Default start time for other days
        except Exception as e:
            print(f"ERROR in _calculate_optimal_start_time: {e}")
            return time(10, 0)  # Fallback to 10 AM on error
            
    def _parse_time(self, time_str: Any) -> time:
        """Parse a time string into a time object with robust error handling.
        
        Args:
            time_str: Time string in various formats (e.g., '14:30', '2:30 PM', '2:30PM', '9:00 AM - 11:00 AM')
                     Can also be a time object or datetime.time object.
                
        Returns:
            A time object with valid hour (0-23) and minute (0-59) values.
            Defaults to 9:00 AM for unparseable times.
        """
        DEFAULT_TIME = time(9, 0)  # Default time to return on error
        
        def safe_int(value: Any, default: int = 0) -> int:
            """Safely convert value to integer with default fallback."""
            try:
                return int(value)
            except (ValueError, TypeError):
                return default
        
        def validate_time(hours: int, minutes: int) -> time:
            """Validate and adjust time values to be within valid ranges."""
            # Ensure hours is between 0 and 23
            hours = max(0, min(23, hours))
            # Ensure minutes is between 0 and 59
            minutes = max(0, min(59, minutes))
            return time(hours, minutes)
        
        # If already a time object, validate and return
        if isinstance(time_str, time):
            return validate_time(time_str.hour, time_str.minute)
            
        # Handle None or empty string
        if not time_str:
            print("WARNING: Empty time input")
            return DEFAULT_TIME
            
        # Convert to string for parsing
        time_str = str(time_str).strip().upper()
        if not time_str:
            print("WARNING: Empty time string after conversion")
            return DEFAULT_TIME
            
        try:
            print(f"DEBUG: Parsing time string: '{time_str}'")
            
            # Handle time ranges by taking the first time
            if '-' in time_str:
                time_str = time_str.split('-')[0].strip()
                print(f"DEBUG: Extracted first time from range: '{time_str}'")
            
            # Remove any non-numeric characters except : and AM/PM
            clean_str = ''.join(c for c in time_str if c.isdigit() or c in (':', 'A', 'P', 'M', ' ')).strip()
            
            # Handle 12-hour format with AM/PM
            is_12h = 'AM' in clean_str or 'PM' in clean_str
            period = ''
            
            if is_12h:
                period = 'AM' if 'AM' in clean_str else 'PM'
                clean_str = clean_str.replace('AM', '').replace('PM', '').strip()
            
            # Handle formats with or without colon
            if ':' in clean_str:
                parts = clean_str.split(':')
                hours = safe_int(parts[0].strip(), 12)  # Default to 12 for invalid hours
                minutes = safe_int(parts[1].strip() if len(parts) > 1 else 0, 0)
            else:
                # Handle formats like '900' or '9' or '9PM'
                digits = ''.join(c for c in clean_str if c.isdigit())
                if len(digits) <= 2:
                    hours = safe_int(digits, 12)  # Default to 12 for invalid hours
                    minutes = 0
                else:
                    # Handle '930' -> 9:30 or '1330' -> 13:30
                    hours = safe_int(digits[:-2] if len(digits) > 2 else digits[:2], 12)
                    minutes = safe_int(digits[-2:], 0)
            
            # Convert 12-hour to 24-hour format if needed
            if is_12h:
                if period == 'PM' and hours < 12:
                    hours += 12
                elif period == 'AM' and hours == 12:
                    hours = 0
            
            print(f"DEBUG: Parsed time - hours: {hours}, minutes: {minutes}")
            
            # Validate and adjust time values
            return validate_time(hours, minutes)
            
        except Exception as e:
            print(f"ERROR: Unexpected error parsing time '{time_str}': {e}")
            import traceback
            traceback.print_exc()
            return DEFAULT_TIME
            
    def _parse_itinerary_response(self, response: Dict, travel_request: TravelPlanRequest) -> TravelItinerary:
        """Parse the LLM response into DailyItinerary objects with proper time slots."""
        print("DEBUG: Starting to parse itinerary response")
        print(f"DEBUG: Full response type: {type(response)}")
        print(f"DEBUG: Full response keys: {list(response.keys())}")
        print(f"DEBUG: Response content (first 1000 chars): {str(response)[:1000]}")
        
        # Initialize empty list to store daily itineraries
        daily_itineraries = []
        current_date = travel_request.start_date if hasattr(travel_request, 'start_date') else date.today()
        
        # Check for different response formats
        if isinstance(response, dict):
            # Format 0: Response has 'trip_name', 'introduction', and 'daily_itinerary' keys
            if all(key in response for key in ['trip_name', 'introduction', 'daily_itinerary']):
                print("DEBUG: Found 'trip_name', 'introduction', and 'daily_itinerary' keys in response")
                
                if not isinstance(response['daily_itinerary'], list):
                    print(f"WARNING: 'daily_itinerary' is not a list: {response['daily_itinerary']}")
                    return daily_itineraries
                
                for day_idx, day_item in enumerate(response['daily_itinerary']):
                    if not isinstance(day_item, dict):
                        print(f"WARNING: Skipping non-dict day item: {day_item}")
                        continue
                        
                    activities = []
                    day_number = day_item.get('day', day_idx + 1)
                    day_intro = day_item.get('day_introduction', f'Day {day_number} of your {travel_request.destination} adventure!')
                    
                    # Handle activities if present
                    if 'activities' in day_item and isinstance(day_item['activities'], list):
                        for activity_item in day_item['activities']:
                            if not isinstance(activity_item, dict):
                                continue
                                
                            activity = {
                                'name': activity_item.get('name', activity_item.get('activity_name', 'Activity')),
                                'start_time': activity_item.get('start_time', activity_item.get('time', '')),
                                'end_time': activity_item.get('end_time', ''),
                                'location': activity_item.get('location', travel_request.destination),
                                'description': activity_item.get('description', activity_item.get('details', ''))
                            }
                            activities.append(activity)
                    
                    daily_itinerary = {
                        'day': day_number,
                        'date': (datetime.datetime.now() + datetime.timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                        'introduction': day_intro,
                        'activities': activities
                    }
                    daily_itineraries.append(daily_itinerary)
                
                print(f"DEBUG: Created {len(daily_itineraries)} daily itineraries from daily_itinerary format")
                return daily_itineraries
            
            # Format 1: Response has 'day_introduction' and 'daily_schedule' keys
            elif 'daily_schedule' in response and isinstance(response['daily_schedule'], list):
                print("DEBUG: Found 'daily_schedule' key in response")
                activities = []
                
                # Process each activity in the daily schedule
                for i, activity_item in enumerate(response['daily_schedule']):
                    if not isinstance(activity_item, dict):
                        print(f"WARNING: Skipping non-dict activity item: {activity_item}")
                        continue
                        
                    # Extract time range from time_slot
                    time_slot = activity_item.get('time_slot', '')
                    time_range = time_slot.split(' - ', 1) if time_slot else []
                    start_time = time_range[0].strip() if len(time_range) > 0 else ""
                    end_time = time_range[1].strip() if len(time_range) > 1 else ""
                    
                    # Create activity dictionary
                    activity = {
                        'name': activity_item.get('activity_name', f'Activity {i+1}'),
                        'start_time': start_time,
                        'end_time': end_time,
                        'location': activity_item.get('location', travel_request.destination),
                        'description': activity_item.get('description', '')
                    }
                    activities.append(activity)
                
                # Create daily itinerary with the introduction if available
                introduction = response.get('day_introduction', f'Day 1 of your {travel_request.destination} adventure!')
                daily_itinerary = {
                    'day': 1,
                    'date': current_date.strftime("%Y-%m-%d"),
                    'introduction': introduction,
                    'activities': activities
                }
                daily_itineraries.append(daily_itinerary)
                print(f"DEBUG: Created daily itinerary with {len(activities)} activities from daily_schedule format")
            
            # Format 1: Response has a 'schedule' key with a list of activities
            elif 'schedule' in response and isinstance(response['schedule'], list):
                print("DEBUG: Found 'schedule' key in response, processing as single day")
                activities = []
                
                # Process each activity in the schedule
                for i, activity_item in enumerate(response['schedule']):
                    if not isinstance(activity_item, dict):
                        print(f"WARNING: Skipping non-dict activity item: {activity_item}")
                        continue
                        
                    # Extract time range (e.g., "8:00 AM - 9:00 AM")
                    time_str = activity_item.get('time', '')
                    if not time_str and 'start_time' in activity_item and 'end_time' in activity_item:
                        # Handle case where start_time and end_time are separate fields
                        start_time = activity_item.get('start_time', '')
                        end_time = activity_item.get('end_time', '')
                    else:
                        # Handle time range string
                        time_range = time_str.split(' - ', 1)
                        start_time = time_range[0].strip() if len(time_range) > 0 else ""
                        end_time = time_range[1].strip() if len(time_range) > 1 else ""
                    
                    # Create activity dictionary
                    activity = {
                        'name': activity_item.get('activity', activity_item.get('name', f'Activity {i+1}')),
                        'start_time': start_time,
                        'end_time': end_time,
                        'location': activity_item.get('location', travel_request.destination),
                        'description': activity_item.get('description', activity_item.get('details', ''))
                    }
                    activities.append(activity)
                
                # Create daily itinerary
                daily_itinerary = {
                    'day': response.get('day', 1),
                    'date': current_date.strftime("%Y-%m-%d"),
                    'introduction': response.get('introduction', f'Day 1 of your {travel_request.destination} adventure!'),
                    'activities': activities
                }
                daily_itineraries.append(daily_itinerary)
                print(f"DEBUG: Created daily itinerary with {len(activities)} activities")
                
            # Format 2: Response has 'days' key with a list of days
            elif 'days' in response and isinstance(response['days'], list):
                print("DEBUG: Found 'days' key with list of days")
                for day_idx, day_data in enumerate(response['days']):
                    if not isinstance(day_data, dict):
                        print(f"WARNING: Skipping non-dict day data: {day_data}")
                        continue
                        
                    activities = []
                    day_num = day_data.get('day', day_idx + 1)
                    
                    # Process activities for this day
                    for i, activity_item in enumerate(day_data.get('activities', [])):
                        if not isinstance(activity_item, dict):
                            print(f"WARNING: Skipping non-dict activity item: {activity_item}")
                            continue
                            
                        # Extract time information
                        time_str = activity_item.get('time', '')
                        if not time_str and 'start_time' in activity_item and 'end_time' in activity_item:
                            start_time = activity_item.get('start_time', '')
                            end_time = activity_item.get('end_time', '')
                        else:
                            time_range = time_str.split(' - ', 1)
                            start_time = time_range[0].strip() if len(time_range) > 0 else ""
                            end_time = time_range[1].strip() if len(time_range) > 1 else ""
                        
                        # Create activity dictionary
                        activity = {
                            'name': activity_item.get('activity', activity_item.get('name', f'Activity {i+1}')),
                            'start_time': start_time,
                            'end_time': end_time,
                            'location': activity_item.get('location', travel_request.destination),
                            'description': activity_item.get('description', activity_item.get('details', ''))
                        }
                        activities.append(activity)
                    
                    # Create daily itinerary
                    day_date = (current_date + timedelta(days=day_idx)).strftime("%Y-%m-%d")
                    daily_itinerary = {
                        'day': day_num,
                        'date': day_date,
                        'introduction': day_data.get('introduction', f'Day {day_num} of your {travel_request.destination} adventure!'),
                        'activities': activities
                    }
                    daily_itineraries.append(daily_itinerary)
                    print(f"DEBUG: Added daily itinerary for day {day_num} with {len(activities)} activities")
            
            # Format 3: Response has day1, day2, etc. keys
            elif any(key.startswith(('day', 'Day')) for key in response.keys() if isinstance(key, str)):
                print("DEBUG: Found dayX keys in response")
                # Sort day keys to maintain order (day1, day2, etc.)
                day_keys = sorted(
                    [k for k in response.keys() if isinstance(k, str) and k.lower().startswith('day')],
                    key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
                )
                
                for day_idx, day_key in enumerate(day_keys):
                    day_data = response[day_key]
                    if not isinstance(day_data, dict):
                        print(f"WARNING: Day data for {day_key} is not a dictionary")
                        continue
                        
                    activities = []
                    day_num = day_idx + 1
                    
                    # Process activities for this day
                    for i, activity_item in enumerate(day_data.get('activities', [])):
                        if not isinstance(activity_item, dict):
                            print(f"WARNING: Skipping non-dict activity item: {activity_item}")
                            continue
                            
                        # Extract time information
                        time_str = activity_item.get('time', '')
                        if not time_str and 'start_time' in activity_item and 'end_time' in activity_item:
                            start_time = activity_item.get('start_time', '')
                            end_time = activity_item.get('end_time', '')
                        else:
                            time_range = time_str.split(' - ', 1) if time_str else []
                            start_time = time_range[0].strip() if len(time_range) > 0 else ""
                            end_time = time_range[1].strip() if len(time_range) > 1 else ""
                        
                        # Create activity dictionary
                        activity = {
                            'name': activity_item.get('activity', activity_item.get('name', f'Activity {i+1}')),
                            'start_time': start_time,
                            'end_time': end_time,
                            'location': activity_item.get('location', travel_request.destination),
                            'description': activity_item.get('description', activity_item.get('details', ''))
                        }
                        activities.append(activity)
                    
                    # Create daily itinerary
                    day_date = (current_date + timedelta(days=day_idx)).strftime("%Y-%m-%d")
                    daily_itinerary = {
                        'day': day_num,
                        'date': day_date,
                        'introduction': day_data.get('introduction', f'Day {day_num} of your {travel_request.destination} adventure!'),
                        'activities': activities
                    }
                    daily_itineraries.append(daily_itinerary)
                    print(f"DEBUG: Added daily itinerary for {day_key} with {len(activities)} activities")
            
            # Format 4: Response is a single day with activities directly in the root
            elif any(key in response for key in ['activities', 'itinerary', 'plan']):
                print("DEBUG: Treating response as a single day with activities in root")
                activities = []
                activities_data = []
                
                # Check for different possible activity list keys
                for key in ['activities', 'itinerary', 'plan']:
                    if key in response and isinstance(response[key], list):
                        activities_data = response[key]
                        print(f"DEBUG: Found activities in '{key}' key")
                        break
                
                # Process activities
                for i, activity_item in enumerate(activities_data):
                    if not isinstance(activity_item, dict):
                        print(f"WARNING: Skipping non-dict activity item: {activity_item}")
                        continue
                        
                    # Extract time information
                    time_str = activity_item.get('time', '')
                    if not time_str and 'start_time' in activity_item and 'end_time' in activity_item:
                        start_time = activity_item.get('start_time', '')
                        end_time = activity_item.get('end_time', '')
                    else:
                        time_range = time_str.split(' - ', 1) if time_str else []
                        start_time = time_range[0].strip() if len(time_range) > 0 else ""
                        end_time = time_range[1].strip() if len(time_range) > 1 else ""
                    
                    # Create activity dictionary
                    activity = {
                        'name': activity_item.get('activity', activity_item.get('name', f'Activity {i+1}')),
                        'start_time': start_time,
                        'end_time': end_time,
                        'location': activity_item.get('location', travel_request.destination),
                        'description': activity_item.get('description', activity_item.get('details', ''))
                    }
                    activities.append(activity)
                
                # Create daily itinerary
                daily_itinerary = {
                    'day': response.get('day', 1),
                    'date': current_date.strftime("%Y-%m-%d"),
                    'introduction': response.get('introduction', f'Day 1 of your {travel_request.destination} adventure!'),
                    'activities': activities
                }
                daily_itineraries.append(daily_itinerary)
                print(f"DEBUG: Created daily itinerary with {len(activities)} activities")
            
            # Format 5: Response is a list of activities
            elif isinstance(response, list):
                print("DEBUG: Response is a list, treating as activities for a single day")
                activities = []
                
                for i, activity_item in enumerate(response):
                    if not isinstance(activity_item, dict):
                        print(f"WARNING: Skipping non-dict activity item: {activity_item}")
                        continue
                        
                    # Extract time information
                    time_str = activity_item.get('time', '')
                    if not time_str and 'start_time' in activity_item and 'end_time' in activity_item:
                        start_time = activity_item.get('start_time', '')
                        end_time = activity_item.get('end_time', '')
                    else:
                        time_range = time_str.split(' - ', 1) if time_str else []
                        start_time = time_range[0].strip() if len(time_range) > 0 else ""
                        end_time = time_range[1].strip() if len(time_range) > 1 else ""
                    
                    # Create activity dictionary
                    activity = {
                        'name': activity_item.get('activity', activity_item.get('name', f'Activity {i+1}')),
                        'start_time': start_time,
                        'end_time': end_time,
                        'location': activity_item.get('location', travel_request.destination),
                        'description': activity_item.get('description', activity_item.get('details', ''))
                    }
                    activities.append(activity)
                
                # Create daily itinerary
                daily_itinerary = {
                    'day': 1,
                    'date': current_date.strftime("%Y-%m-%d"),
                    'introduction': f'Day 1 of your {travel_request.destination} adventure!',
                    'activities': activities
                }
                daily_itineraries.append(daily_itinerary)
                print(f"DEBUG: Created daily itinerary with {len(activities)} activities")
            
            else:
                print("WARNING: Unrecognized response format, attempting to extract activities")
                # Try to extract activities from the response
                activities = []
                
                # Check if the response itself is an activity
                if all(k in response for k in ['activity', 'time']) or all(k in response for k in ['name', 'start_time']):
                    # Single activity in the root
                    activity = {
                        'name': response.get('activity', response.get('name', 'Activity')),
                        'start_time': response.get('start_time', ''),
                        'end_time': response.get('end_time', ''),
                        'location': response.get('location', travel_request.destination),
                        'description': response.get('description', response.get('details', ''))
                    }
                    activities.append(activity)
                
                # Create daily itinerary if we found any activities
                if activities:
                    daily_itinerary = {
                        'day': 1,
                        'date': current_date.strftime("%Y-%m-%d"),
                        'introduction': f'Day 1 of your {travel_request.destination} adventure!',
                        'activities': activities
                    }
                    daily_itineraries.append(daily_itinerary)
                    print(f"DEBUG: Created daily itinerary with {len(activities)} activities from root level")
                else:
                    print("WARNING: Could not extract activities from response, using fallback")
                    return self._create_basic_itinerary(travel_request, [])
        else:
            print("WARNING: Response is not a dictionary, using fallback")
            return self._create_basic_itinerary(travel_request, [])
        
        # If response is a string, try to parse it as JSON
        if isinstance(response, str):
            try:
                import json
                response = json.loads(response)
                print("DEBUG: Successfully parsed string response as JSON")
            except json.JSONDecodeError:
                print("WARNING: Response is a string but not valid JSON")
                return self._create_basic_itinerary(travel_request, [])
        
        # Initialize empty list to store daily itineraries
        daily_itineraries = []
        
        # Track the current date for multi-day itineraries
        current_date = getattr(travel_request, 'start_date', date.today())
        
        try:
            print(f"DEBUG: Response type: {type(response)}")
            
            # Handle different response formats
            if not response:
                print("WARNING: Empty response from LLM")
                return self._create_basic_itinerary(travel_request, [])
            
            # Initialize days_data list to hold all day entries
            days_data = []
            
            # Case 1: Response has a 'days' key with a list of days
            if 'days' in response and isinstance(response['days'], list):
                print("DEBUG: Found 'days' key with list of days")
                days_data = response['days']
            # Case 2: Response has day1, day2, etc. keys
            elif any(key.startswith(('day', 'Day')) for key in response.keys() if isinstance(key, str)):
                print("DEBUG: Found dayX keys in response")
                # Sort day keys to maintain order (day1, day2, etc.)
                day_keys = sorted(
                    [k for k in response.keys() if isinstance(k, str) and k.lower().startswith('day')],
                    key=lambda x: int(''.join(filter(str.isdigit, x)) or 0)
                )
                days_data = [response[key] for key in day_keys if isinstance(response[key], (dict, list))]
            # Case 3: Response is a dictionary that might be a single day
            elif isinstance(response, dict) and any(key in response for key in ['schedule', 'activities', 'itinerary']):
                print("DEBUG: Treating response as a single day")
                days_data = [response]
            # Case 4: Response is a list (assume list of days)
            elif isinstance(response, list):
                print(f"DEBUG: Response is a list with {len(response)} items")
                days_data = response
            else:
                print(f"WARNING: Unrecognized response format, attempting to process as single day")
                days_data = [response]
            
            # Ensure days_data is a list
            if not isinstance(days_data, list):
                print(f"WARNING: days_data is not a list, converting: {type(days_data)}")
                days_data = [days_data]
            
            print(f"DEBUG: Processing {len(days_data)} days of itinerary")
            
            for day_idx, day_data in enumerate(days_data):
                if not day_data:
                    print(f"WARNING: Empty day_data for day {day_idx+1}")
                    continue
                
                # Calculate optimal start time based on day and trip context
                try:
                    current_time = self._calculate_optimal_start_time(travel_request, day_num)
                    print(f"DEBUG: Day {day_num} - Context-aware start time: {current_time}")
                except Exception as e:
                    print(f"WARNING: Error calculating optimal start time, using default 9:00 AM. Error: {e}")
                    current_time = time(9, 0)
                    
                # Create activities from activities_data
                activities = []
                for i, activity_data in enumerate(activities_data):
                    if not isinstance(activity_data, dict):
                        continue
                        
                    # Create activity with proper error handling
                    try:
                        activity = {
                            'name': activity_data.get('name', activity_data.get('activity', f'Activity {i+1}')),
                            'start_time': activity_data.get('start_time', ''),
                            'end_time': activity_data.get('end_time', ''),
                            'location': activity_data.get('location', travel_request.destination),
                            'description': activity_data.get('description', '')
                        }
                        activities.append(activity)
                    except Exception as e:
                        print(f"WARNING: Error creating activity {i+1}: {e}")
                        continue
                        
                # Create daily itinerary
                daily_itinerary = {
                    'day': day_num,
                    'date': (current_date + timedelta(days=day_idx)).strftime("%Y-%m-%d"),
                    'activities': activities
                }
                daily_itineraries.append(daily_itinerary)
                print(f"DEBUG: Added daily itinerary for day {day_num} with {len(activities)} activities")
                
                # Add arrival activity for the first day
                if day_idx == 0 and travel_request.origin and travel_request.destination:
                    try:
                        print("DEBUG: Adding arrival activity for first day")
                        # Ensure we have a valid start time
                        start_time = current_time if isinstance(current_time, time) else time(9, 0)
                        
                        # Ensure start_time is a valid time object
                        if not hasattr(start_time, 'hour') or not hasattr(start_time, 'minute'):
                            print("WARNING: Invalid start_time, using 9:00 AM")
                            start_time = time(9, 0)
                        
                        # Calculate end time (1 hour after start)
                        end_time = self._add_minutes(start_time, 60)
                        
                        # Ensure end time is valid
                        if not hasattr(end_time, 'hour') or not hasattr(end_time, 'minute'):
                            print("WARNING: Invalid end_time, using 10:00 AM")
                            end_time = time(10, 0)
                        
                        # Create arrival activity with proper time formatting
                        arrival_activity = {
                            'name': f"Arrive in {travel_request.destination}",
                            'start_time': start_time.strftime("%H:%M"),
                            'end_time': end_time.strftime("%H:%M"),
                            'location': f"{travel_request.destination} Airport/Station",
                            'description': f"Arrival in {travel_request.destination}. Collect your luggage and proceed to your accommodation."
                        }
                        
                        # Add to activities and update current time
                        activities.append(arrival_activity)
                        current_time = self._parse_time(arrival_activity['end_time'])
                        print(f"DEBUG: After arrival activity, current time: {current_time}")
                    except Exception as e:
                        print(f"ERROR: Failed to add arrival activity: {e}")
                        # Continue without arrival activity if there's an error
                
                # Process each activity in the day
                for j, activity_data in enumerate(activities_data):
                    try:
                        print(f"\nDEBUG: Processing activity {j+1} of {len(activities_data)}")
                        if isinstance(activity_data, dict):
                            print(f"DEBUG: Activity data keys: {list(activity_data.keys())}")
                        
                        # Initialize activity with default values
                        activity = {
                            'name': str(activity_data.get('activity', activity_data.get('name', f'Activity {j+1}'))),
                            'location': str(activity_data.get('location', '')),
                            'description': str(activity_data.get('description', activity_data.get('details', ''))),
                            'estimated_cost': str(activity_data.get('estimated_cost', activity_data.get('cost', '')))
                        }
                        
                        print(f"DEBUG: Activity name: {activity['name']}")
                        
                        # Parse duration from activity data with fallback to 60 minutes
                        duration_minutes = 60  # Default duration
                        try:
                            # Try to get duration from various possible fields
                            if 'duration_minutes' in activity_data and activity_data['duration_minutes'] is not None:
                                duration_minutes = int(activity_data['duration_minutes'])
                            elif 'duration' in activity_data and activity_data['duration']:
                                duration_str = str(activity_data['duration']).lower()
                                if 'hour' in duration_str or 'hr' in duration_str:
                                    # Extract hours and convert to minutes
                                    duration_parts = duration_str.split()
                                    hours = float(duration_parts[0]) if duration_parts else 1.0
                                    duration_minutes = int(hours * 60)
                                elif 'min' in duration_str:
                                    # Extract minutes
                                    duration_parts = duration_str.split()
                                    duration_minutes = int(duration_parts[0]) if duration_parts else 60
                                else:
                                    # Try to parse as a number (assume minutes)
                                    duration_minutes = int(float(duration_str))
                        except (ValueError, TypeError) as e:
                            print(f"WARNING: Error parsing duration, using default 60 minutes: {e}")
                            duration_minutes = 60
                        
                        # Ensure duration is within reasonable bounds (5 min to 12 hours)
                        duration_minutes = max(5, min(720, duration_minutes))
                        print(f"DEBUG: Using duration: {duration_minutes} minutes")
                        
                        try:
                            start_time = self._ensure_valid_time(current_time)
                            end_time = self._calculate_next_activity_time(start_time, duration_minutes)
                            print(f"DEBUG: Initial times - start: {start_time}, end: {end_time}")
                        except Exception as e:
                            print(f"WARNING: Error initializing default times: {e}")
                            start_time = time(9, 0)
                            end_time = time(10, 0)
                        
                        # Check for explicit time range in the activity data
                        time_range = activity_data.get('time', '')
                        if time_range and isinstance(time_range, str) and '-' in time_range:
                            print(f"DEBUG: Found time range: {time_range}")
                            try:
                                # Clean and parse the time range
                                time_range = time_range.strip()
                                time_parts = [t.strip() for t in time_range.split('-', 1)]
                                
                                if len(time_parts) == 2:
                                    start_str, end_str = time_parts
                                    print(f"DEBUG: Parsing times - start: '{start_str}', end: '{end_str}'")
                                    
                                    # Parse the times with robust error handling
                                    parsed_start = self._parse_time(start_str)
                                    parsed_end = self._parse_time(end_str)
                                    
                                    # Only use parsed times if they're valid
                                    if isinstance(parsed_start, time):
                                        start_time = self._ensure_valid_time(parsed_start)
                                        print(f"DEBUG: Using parsed start time: {start_time}")
                                    
                                    if isinstance(parsed_end, time):
                                        end_time = self._ensure_valid_time(parsed_end)
                                        print(f"DEBUG: Using parsed end time: {end_time}")
                                    
                                    # Calculate duration from the parsed times if both are valid
                                    if hasattr(start_time, 'hour') and hasattr(end_time, 'hour'):
                                        start_total = start_time.hour * 60 + start_time.minute
                                        end_total = end_time.hour * 60 + end_time.minute
                                        
                                        if end_total > start_total:
                                            duration_minutes = end_total - start_total
                                            print(f"DEBUG: Calculated duration from time range: {duration_minutes} minutes")
                                        else:
                                            print(f"WARNING: End time {end_time} is not after start time {start_time}")
                                            end_time = self._calculate_next_activity_time(start_time, duration_minutes)
                                    
                                    print(f"DEBUG: Final times after parsing - start: {start_time}, end: {end_time}")
                                
                            except Exception as e:
                                print(f"WARNING: Error parsing time range: {e}")
                                # Fall back to default times
                        
                        # Final validation of times
                        try:
                            # Ensure times are valid time objects
                            if not hasattr(start_time, 'hour') or not hasattr(start_time, 'minute'):
                                print("WARNING: Invalid start_time, using current_time or default")
                                start_time = self._ensure_valid_time(current_time if hasattr(current_time, 'hour') else None)
                            
                            if not hasattr(end_time, 'hour') or not hasattr(end_time, 'minute'):
                                print("WARNING: Invalid end_time, calculating from start_time")
                                end_time = self._calculate_next_activity_time(start_time, duration_minutes)
                            
                            if not hasattr(current_time, 'hour') or not hasattr(current_time, 'minute'):
                                print("WARNING: Invalid current_time, using default 4 PM")
                                current_time = time(16, 0)  # Default to 4 PM if current_time is invalid
                            
                            # Ensure end time is after start time
                            start_total = start_time.hour * 60 + start_time.minute
                            end_total = end_time.hour * 60 + end_time.minute
                            
                            if end_total <= start_total:
                                print(f"WARNING: End time {end_time} is not after start time {start_time}")
                                end_time = self._calculate_next_activity_time(start_time, duration_minutes)
                                print(f"DEBUG: Adjusted end time to: {end_time}")
                            
                            print(f"DEBUG: Activity {j+1} - {start_time} to {end_time} (duration: {duration_minutes} min)")
                            
                        except Exception as e:
                            print(f"ERROR: Failed to validate times: {e}")
                            # Fall back to safe defaults
                            start_time = time(9, 0)
                            end_time = time(10, 0)
                            duration_minutes = 60
                        
                        # Set the final times in the activity
                        activity['start_time'] = start_time.strftime("%H:%M")
                        activity['end_time'] = end_time.strftime("%H:%M")
                        
                        # Add travel time if specified
                        travel_time = activity_data.get('travel_time', '')
                        if travel_time and isinstance(travel_time, str) and 'min' in travel_time.lower():
                            try:
                                travel_mins = int(''.join(filter(str.isdigit, travel_time)) or '0')
                                if travel_mins > 0:
                                    activity['travel_time'] = f"{travel_mins} min"
                            except (ValueError, TypeError) as e:
                                print(f"WARNING: Could not parse travel time: {travel_time}. Error: {e}")
                        
                        # Add the activity to the list
                        activities.append(activity)
                        print(f"DEBUG: Added activity: {activity['name']} ({activity['start_time']} - {activity['end_time']})")
                        
                        # Update current time for the next activity with a buffer
                        current_time = self._add_minutes(end_time, 15)  # 15-minute buffer between activities
                        
                    except Exception as e:
                        print(f"ERROR: Failed to process activity {j+1}: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue with the next activity if one fails
                
                # Add departure activity for the last day if it's a round trip
                if day_idx == len(days_data) - 1 and travel_request.origin and travel_request.destination and travel_request.origin != travel_request.destination:
                    try:
                        print("DEBUG: Adding return journey activities")
                        
                        # Ensure we have a valid current_time before calculating departure
                        if not hasattr(current_time, 'hour') or not hasattr(current_time, 'minute'):
                            print("WARNING: Invalid current_time, using default 4 PM")
                            current_time = time(16, 0)  # Default to 4 PM if current_time is invalid
                        
                        # Add travel time to return to starting point (2 hours before departure)
                        departure_time = self._add_minutes(current_time, 30)  # 30 min buffer
                        
                        # Add return journey activity
                        return_activity = {
                            'name': f"Return to {travel_request.origin}",
                            'start_time': departure_time.strftime("%H:%M"),
                            'end_time': (datetime.combine(date.today(), departure_time) + timedelta(hours=2)).time().strftime("%H:%M"),
                            'location': f"{travel_request.destination} to {travel_request.origin}",
                            'description': f"Return journey from {travel_request.destination} back to {travel_request.origin}."
                        }
                        
                        # Add to activities
                        activities.append(return_activity)
                        print(f"DEBUG: Added return journey activity: {return_activity['name']}")
                        
                        # Update current time after return journey
                        current_time = self._parse_time(return_activity['end_time'])
                        
                    except Exception as e:
                        print(f"ERROR: Failed to add return journey activity: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Create the daily itinerary for this day
                try:
                    # Ensure we have a valid date for the daily itinerary
                    day_date = current_date + timedelta(days=day_idx) if hasattr(current_date, 'day') else date.today()
                    
                    # Create the daily itinerary
                    daily_itinerary = {
                        'day': day_num,
                        'date': day_date.strftime("%Y-%m-%d"),
                        'activities': activities
                    }
                    
                    print(f"DEBUG: Created daily itinerary for day {day_num} with {len(activities)} activities")
                    daily_itineraries.append(daily_itinerary)
                    
                except Exception as e:
                    print(f"ERROR: Failed to create daily itinerary for day {day_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Continue with the next day even if one day fails
            
            # Create the final travel itinerary
            try:
                # Ensure we have at least one daily itinerary
                if not daily_itineraries:
                    print("WARNING: No daily itineraries created, using fallback")
                    return self._create_basic_itinerary(travel_request, [])
                
                # Create the final travel itinerary
                itinerary = {
                    'destination': travel_request.destination,
                    'start_date': daily_itineraries[0]['date'],
                    'end_date': daily_itineraries[-1]['date'],
                    'days': daily_itineraries
                }
                
                print(f"DEBUG: Created travel itinerary with {len(daily_itineraries)} days")
                return TravelItinerary(**itinerary)
                
            except Exception as e:
                print(f"ERROR: Failed to create final travel itinerary: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to basic itinerary
                return self._create_basic_itinerary(travel_request, [])
                
        except Exception as e:
            print(f"CRITICAL: Unhandled error in _parse_itinerary_response: {e}")
            import traceback
            traceback.print_exc()
            # Return a basic itinerary as a last resort
            return self._create_basic_itinerary(travel_request, [])
        
        # If we reach here, return an empty itinerary as a last resort
        return TravelItinerary(
            destination=travel_request.destination,
            duration_days=1,
            daily_plans=[]
        )

    def _parse_time(self, time_str: Any) -> time:
        """Parse a time string into a time object with robust error handling.
        
        Args:
            time_str: Time string in various formats (e.g., '14:30', '2:30 PM', '2:30PM', '9:00 AM - 11:00 AM')
                     Can also be a time object or datetime.time object.
            
        Returns:
            A time object with valid hour (0-23) and minute (0-59) values.
            Defaults to 9:00 AM for unparseable times.
        """
        DEFAULT_TIME = time(9, 0)  # Default time to return on error
        
        # Handle None or empty input
        if time_str is None:
            print("WARNING: None time provided, using default 9:00 AM")
            return DEFAULT_TIME
            
        # If already a time object, validate and return
        if isinstance(time_str, time):
            try:
                # Ensure valid hour and minute values
                return time(
                    max(0, min(23, time_str.hour)),
                    max(0, min(59, getattr(time_str, 'minute', 0)))
                )
            except (ValueError, AttributeError) as e:
                print(f"WARNING: Invalid time object: {time_str}, error: {e}")
                return DEFAULT_TIME
        
        # Convert to string if not already
        time_str = str(time_str).strip()
        if not time_str:
            print("WARNING: Empty time string provided, using 9:00 AM as fallback")
            return DEFAULT_TIME
        
        print(f"DEBUG: Parsing time string: '{time_str}'")
            
        def safe_int(value: str, default: int = 0) -> int:
            """Safely convert string to integer with default fallback."""
            try:
                return int(str(value).strip())
            except (ValueError, TypeError, AttributeError):
                return default
        
        # Handle time ranges (e.g., '9:00 AM - 11:00 AM' or '14:00-16:00')
        if '-' in time_str:
            print(f"DEBUG: Extracting start time from range: {time_str}")
            time_str = time_str.split('-', 1)[0].strip()
        
        # Try standard formats first
        time_formats = [
            "%H:%M", "%I:%M %p", "%I:%M%p",  # Standard formats
            "%H:%M:%S", "%I:%M:%S %p", "%I:%M:%S%p",  # With seconds
            "%I %p", "%H"  # Just hour and period, or just hour (24h or 12h)
        ]
        
        for fmt in time_formats:
            try:
                dt = datetime.strptime(time_str, fmt)
                result = time(
                    max(0, min(23, dt.hour)),
                    max(0, min(59, getattr(dt, 'minute', 0)))
                )
                print(f"DEBUG: Successfully parsed with format '{fmt}': {result}")
                return result
            except ValueError:
                continue
        
        # Handle 12-hour format with AM/PM (more flexible handling)
        time_str_upper = time_str.upper()
        if 'AM' in time_str_upper or 'PM' in time_str_upper:
            try:
                # Clean up the string
                time_str_clean = time_str_upper.replace(' ', '')
                
                # Handle formats like '2:30PM' or '11:45 AM'
                if 'AM' in time_str_clean:
                    time_part = time_str_clean.split('AM', 1)[0]
                    is_pm = False
                else:
                    time_part = time_str_clean.split('PM', 1)[0]
                    is_pm = True
                
                # Parse hours and minutes
                if ':' in time_part:
                    parts = time_part.split(':')
                    hours = safe_int(parts[0], 0)
                    minutes = safe_int(parts[1], 0) if len(parts) > 1 else 0
                else:
                    hours = safe_int(time_part, 0)
                    minutes = 0
                
                # Convert 12-hour to 24-hour format
                if is_pm and hours < 12:
                    hours += 12
                elif not is_pm and hours == 12:  # 12:00 AM is 00:00
                    hours = 0
                
                # Ensure valid time values
                hours = max(0, min(23, hours))
                minutes = max(0, min(59, minutes))
                
                result = time(hours, minutes)
                print(f"DEBUG: Successfully parsed 12-hour time: {result}")
                return result
                
            except Exception as e:
                print(f"Warning: Error parsing 12-hour time '{time_str}': {e}")
        
        # Handle 24-hour format (e.g., '14:30' or '14:30:00' or '14' or '14:30:45.123')
        try:
            # Clean the string and split by non-digits
            clean_str = ''.join(c if c.isdigit() or c == ':' else ' ' for c in time_str)
            parts = [p for p in clean_str.split(':') if p.strip()]
            
            if parts:
                hours = safe_int(parts[0], 0)
                minutes = safe_int(parts[1], 0) if len(parts) > 1 else 0
                
                # Ensure valid time values
                hours = max(0, min(23, hours))
                minutes = max(0, min(59, minutes))
                
                result = time(hours, minutes)
                print(f"DEBUG: Successfully parsed 24-hour time: {result}")
                return result
                
        except Exception as e:
            print(f"Warning: Error parsing 24-hour time '{time_str}': {e}")
        
        # If we get here, all parsing attempts failed
        print(f"WARNING: Could not parse time string: '{time_str}'. Using default 9:00 AM")
        return DEFAULT_TIME
    
    async def _create_basic_itinerary(
        self, 
        travel_request: TravelPlanRequest, 
        selected_pois: List[Dict[str, Any]]
    ) -> List[DailyItinerary]:
        """Create a basic itinerary as a fallback."""
        daily_itineraries = []
        start_date = getattr(travel_request, 'start_date', date.today())
        duration_days = getattr(travel_request, 'duration_days', 1)
        
        # Debug: Print the type and content of selected_pois
        print(f"\n{'='*80}\nDEBUG: _create_basic_itinerary - START\n{'='*80}")
        print(f"DEBUG: selected_pois type: {type(selected_pois)}")
        print(f"DEBUG: selected_pois length: {len(selected_pois) if selected_pois else 0}")
        
        if selected_pois:
            print(f"\nDEBUG: First POI details:")
            print(f"  - Type: {type(selected_pois[0])}")
            if hasattr(selected_pois[0], '__dict__'):
                print(f"  - Attributes: {dir(selected_pois[0])}")
                if hasattr(selected_pois[0], 'name'):
                    print(f"  - Name: {selected_pois[0].name}")
                if hasattr(selected_pois[0], 'duration_minutes'):
                    print(f"  - Duration: {selected_pois[0].duration_minutes} mins")
            elif isinstance(selected_pois[0], dict):
                print(f"  - Dictionary keys: {selected_pois[0].keys()}")
                if 'poi' in selected_pois[0]:
                    print(f"  - 'poi' key exists, type: {type(selected_pois[0]['poi'])}")
                    if hasattr(selected_pois[0]['poi'], '__dict__'):
                        print(f"  - POI object attributes: {dir(selected_pois[0]['poi'])}")
            
            # Print all POI names for verification
            print("\nDEBUG: All POI names:")
            for i, poi in enumerate(selected_pois[:5]):  # Limit to first 5 to avoid too much output
                name = "Unknown"
                if hasattr(poi, 'name'):
                    name = poi.name
                elif isinstance(poi, dict):
                    if 'name' in poi:
                        name = poi['name']
                    elif 'poi' in poi and hasattr(poi['poi'], 'name'):
                        name = poi['poi'].name
                print(f"  {i+1}. {name}")
        
        # Ensure we have a list of POIs (handle case where it might be None or empty)
        if not selected_pois:
            print("\nWARNING: No POIs provided to _create_basic_itinerary")
            selected_pois = []
        else:
            print(f"\nDEBUG: Processing {len(selected_pois)} POIs for {duration_days} days")
        
        # Distribute POIs across days, ensuring at least 2-3 POIs per day
        pois_per_day = max(2, min(4, len(selected_pois) // max(1, duration_days))) if selected_pois else 0
        if pois_per_day == 0 and selected_pois:  # If we have fewer POIs than days
            pois_per_day = 1
            
        for day in range(1, duration_days + 1):
            current_time = self._calculate_optimal_start_time(travel_request, day)
            activities = []
            
            # Calculate POIs for this day
            start_idx = (day - 1) * pois_per_day
            end_idx = min(start_idx + pois_per_day, len(selected_pois))
            day_pois = selected_pois[start_idx:end_idx]
            
            # If we're on the last day and have remaining POIs, include them
            if day == duration_days and end_idx < len(selected_pois):
                day_pois = selected_pois[start_idx:]
            
            print(f"\nDEBUG: Day {day} - Processing {len(day_pois)} POIs (indices {start_idx} to {end_idx-1})")
            
            if not day_pois:
                # If no POIs for this day, add a free day
                activities = [
                    {
                        'name': "Free Day",
                        'start_time': "09:00",
                        'end_time': "17:00",
                        'location': travel_request.destination,
                        'description': "Free time to explore at your own pace"
                    }
                ]
                print(f"\nDEBUG: Added Free Day for day {day} (no POIs in day_pois)")
                print(f"DEBUG: start_idx: {start_idx}, end_idx: {end_idx}, pois_per_day: {pois_per_day}, total_pois: {len(selected_pois)}")
                print(f"DEBUG: selected_pois is empty: {not bool(selected_pois)}")
            else:
                print(f"\nDEBUG: Processing {len(day_pois)} POIs for day {day}")
                for i, poi in enumerate(day_pois):
                    name = "Unknown"
                    if hasattr(poi, 'name'):
                        name = poi.name
                    elif isinstance(poi, dict):
                        if 'name' in poi:
                            name = poi['name']
                        elif 'poi' in poi and hasattr(poi['poi'], 'name'):
                            name = poi['poi'].name
                    print(f"  POI {i+1}: {name}")
                current_time = time(9, 0)  # Start at 9 AM
                lunch_time = time(12, 0)  # Lunch at 12 PM
                lunch_duration = 60  # 1 hour for lunch
                lunch_end = time(lunch_time.hour, (lunch_time.minute + lunch_duration) % 60)
                
                for i, item in enumerate(day_pois):
                    # Handle different POI formats
                    if isinstance(item, dict):
                        # Handle case where POI is in a 'poi' key
                        poi_data = item.get('poi', item)
                        if not isinstance(poi_data, dict):
                            print(f"WARNING: Invalid POI format at index {i}: {item}")
                            continue
                    else:
                        # Handle case where item is already a POI object
                        poi_data = item.__dict__ if hasattr(item, '__dict__') else {}
                    
                    # Extract POI details with safe defaults
                    name = str(poi_data.get('name', f'Activity {i+1}'))
                    location = str(poi_data.get('location', travel_request.destination))
                    description = str(poi_data.get('description', f'Enjoy {name}'))
                    duration = int(poi_data.get('duration_minutes', 60))
                    
                    # Add travel time (30 minutes between activities, except first)
                    if i > 0:
                        travel_time = 30
                        end_time = time(
                            (current_time.hour + (current_time.minute + travel_time) // 60) % 24,
                            (current_time.minute + travel_time) % 60
                        )
                        travel_activity = {
                            'name': "Travel to next location",
                            'start_time': current_time.strftime("%H:%M"),
                            'end_time': end_time.strftime("%H:%M"),
                            'location': f"Traveling to {location}",
                            'description': f"Travel time to {name}"
                        }
                        activities.append(travel_activity)
                        current_time = end_time
                    
                    # Add the POI activity
                    end_time = time(
                        (current_time.hour + (current_time.minute + duration) // 60) % 24,
                        (current_time.minute + duration) % 60
                    )
                    
                    activity = {
                        'name': name,
                        'start_time': current_time.strftime("%H:%M"),
                        'end_time': end_time.strftime("%H:%M"),
                        'location': location,
                        'description': description
                    }
                    activities.append(activity)
                    current_time = end_time
                    
                    # Add lunch break if it's after 12 PM and before 1 PM
                    if current_time >= time(12, 0) and current_time < time(13, 0) and i > 0:
                        lunch_activity = {
                            'name': "Lunch Break",
                            'start_time': lunch_time.strftime("%H:%M"),
                            'end_time': lunch_end.strftime("%H:%M"),
                            'location': "Local Restaurant",
                            'description': "Time for lunch and relaxation"
                        }
                        activities.append(lunch_activity)
                        current_time = lunch_end
                current_time = lunch_end
            
            # Convert Activity objects to dictionaries and add them to the daily itinerary
            daily_activities = []
            for a in activities:
                if hasattr(a, 'dict'):
                    daily_activities.append(a.dict())
                else:
                    daily_activities.append(a)
            
            # Create the daily itinerary with the converted activities
            daily_itinerary = DailyItinerary(
                day=day,
                date=(start_date + timedelta(days=day-1)).strftime("%Y-%m-%d") if hasattr(travel_request, 'start_date') and travel_request.start_date else "",
                activities=daily_activities
            )
            
            print(f"\nDEBUG: Created daily itinerary for day {day} with {len(daily_activities)} activities")
            for i, activity in enumerate(daily_activities):
                print(f"  Activity {i+1}: {activity.get('name', 'Unnamed')} "
                      f"({activity.get('start_time', '?')} - {activity.get('end_time', '?')})")
            
            daily_itineraries.append(daily_itinerary)
        
        print(f"\n{'='*80}\nDEBUG: _create_basic_itinerary - END\n{'='*80}\n")
            
        return daily_itineraries
    
    def _format_pois_for_prompt(self, pois: List[PointOfInterest]) -> str:
        """Format POIs with rich, engaging descriptions to excite travelers."""
        if not pois:
            return "No specific points of interest selected, but don't worry! We'll create an amazing itinerary based on your interests."
            
        result = []
        for i, poi in enumerate(pois, 1):
            # Extract POI details with fallbacks
            name = getattr(poi, 'name', 'A wonderful location')
            category = getattr(poi, 'category', 'attraction').lower()
            description = getattr(poi, 'description', f'A must-visit {category} that will take your breath away')
            duration = getattr(poi, 'duration_minutes', 60)
            location = getattr(poi, 'location', 'a beautiful location')
            tags = getattr(poi, 'tags', [])
            
            # Generate engaging content based on POI type
            experience_tip = self._generate_experience_tip(name, category, description)
            photo_tip = self._generate_photo_tip(name, category)
            local_insight = self._generate_local_insight(name, category, location)
            
            # Create a rich, engaging POI description
            poi_str = (
                f"## {name.upper()} - {category.upper()} \n\n"
                f"{description.capitalize()}. {experience_tip}\n\n"
                f" **Photo Ops:** {photo_tip}\n\n"
                f" **Local Secrets:** {local_insight}\n\n"
                f" **Suggested Duration:** {duration} minutes\n"
                f" **Location:** {location}\n"
            )
            
            # Add tags if available
            if tags:
                poi_str += f" **Tags:** {', '.join(tags)}\n"
                
            # Add a pro tip based on the category
            poi_str += f" **Pro Tip:** {self._generate_pro_tip(category, name)}\n"
            
            result.append(poi_str)
            
        return "\n---\n".join(result)
    
    def _generate_experience_tip(self, name: str, category: str, description: str) -> str:
        """Generate an engaging experience tip for a POI."""
        import random
        tips = [
            f"As you explore {name}, take a moment to soak in the atmosphere and appreciate the unique details.",
            f"Don't miss the chance to discover hidden corners and local traditions at {name}.",
            f"For the best experience, arrive early to beat the crowds and enjoy {name} at its most peaceful.",
            f"Make sure to take your time at {name} - some of the best experiences happen when you slow down and observe."
        ]
        
        if 'beach' in category:
            tips.extend([
                f"The best time to visit {name} is early morning for peaceful solitude or late afternoon for stunning sunsets.",
                f"Don't forget to bring sunscreen and plenty of water for your time at {name}."
            ])
        elif 'museum' in category or 'gallery' in category:
            tips.extend([
                f"Check if {name} offers any guided tours or audio guides to enhance your visit.",
                f"Plan to spend about {random.randint(30, 90)} minutes at {name} to fully appreciate the exhibits."
            ])
        elif 'hiking' in category or 'nature' in category:
            tips.extend([
                f"Wear comfortable walking shoes and bring water for your adventure at {name}.",
                f"The trails at {name} offer breathtaking views - don't forget your camera!"
            ])
            
        return random.choice(tips)
    
    def _generate_photo_tip(self, name: str, category: str) -> str:
        """Generate a photo tip for a POI."""
        import random
        tips = [
            f"Capture the perfect shot from different angles to showcase the beauty of {name}.",
            f"Look for interesting patterns, textures, and colors that make {name} unique.",
            f"Try shooting during golden hour for magical lighting at {name}.",
            f"Include people in your photos to show the scale and atmosphere of {name}."
        ]
        
        if 'viewpoint' in category or 'scenic' in category:
            tips.append(f"Use a wide-angle lens to capture the breathtaking panorama from {name}.")
        elif 'market' in category or 'street' in category:
            tips.append(f"Capture the vibrant energy and local life at {name} with candid shots.")
            
        return random.choice(tips)
    
    def _generate_local_insight(self, name: str, category: str, location: str) -> str:
        """Generate a local insight for a POI."""
        import random
        insights = [
            f"Locals love {name} for its authentic atmosphere and unique character.",
            f"The area around {name} is full of hidden gems - take some time to explore the side streets.",
            f"Many visitors miss the best part of {name} - make sure to [ask a local about their favorite spot].",
            f"The history of {name} is fascinating - consider joining a guided tour to learn more."
        ]
        
        if 'temple' in category or 'religious' in category:
            insights.append(f"Remember to dress modestly when visiting {name} as a sign of respect.")
        elif 'food' in category or 'restaurant' in category:
            dish_type = random.choice(["chef's special", "local favorite", "seasonal dish"])
            insights.append(f"The {dish_type} at {name} is a must-try!")
            
        return random.choice(insights)
    
    def _generate_pro_tip(self, category: str, name: str) -> str:
        """Generate a pro tip for visiting a POI."""
        import random
        tips = {
            'beach': f"Arrive early to secure a good spot and enjoy the peaceful morning atmosphere at {name}.",
            'museum': f"Check {name}'s website for free admission days or discounted hours.",
            'market': f"Bargaining is expected at {name} - start at about 60% of the asking price.",
            'hiking': f"Wear proper footwear and bring enough water for your adventure at {name}.",
            'restaurant': f"Make reservations in advance for {name}, especially on weekends and holidays.",
            'default': f"Visit {name} during weekdays to avoid the weekend crowds and have a more relaxed experience."
        }
        
        # Check for category matches
        for key, tip in tips.items():
            if key in category:
                return tip
                
        return tips['default']
    
    async def process(self, travel_request: TravelPlanRequest, selected_pois: List[Dict[str, Any]]) -> TravelItinerary:
        """Process the travel request and selected POIs to create a travel itinerary.
        
        Args:
            travel_request: The travel plan request containing destination, duration, etc.
            selected_pois: List of selected points of interest with their metadata.
            
        Returns:
            A TravelItinerary object containing the daily schedules.
        """
        try:
            # Try to generate a detailed itinerary using the LLM
            daily_plans = await self._generate_with_llm(travel_request, selected_pois)
            
            # If no plans were generated, create a basic itinerary
            if not daily_plans:
                daily_plans = self._create_basic_itinerary(travel_request, selected_pois)
            
            # Ensure we have a list of DailyItinerary objects
            if not isinstance(daily_plans, list):
                daily_plans = [daily_plans]
                
            # Convert any dicts to DailyItinerary objects
            for i, plan in enumerate(daily_plans):
                if isinstance(plan, dict):
                    daily_plans[i] = DailyItinerary(**plan)
            
            # Create a TravelItinerary with the daily plans
            return TravelItinerary(
                destination=travel_request.destination,
                duration_days=travel_request.duration_days,
                daily_plans=daily_plans
            )
            
        except Exception as e:
            print(f"Error in process: {e}")
            # Fall back to creating a basic itinerary if there's an error
            daily_plans = self._create_basic_itinerary(travel_request, selected_pois)
            return TravelItinerary(
                destination=travel_request.destination,
                duration_days=travel_request.duration_days,
                daily_plans=daily_plans if isinstance(daily_plans, list) else [daily_plans]
            )
    
    def _add_minutes(self, t: time, minutes: int) -> time:
        """Add minutes to a time object with validation."""
        try:
            if not isinstance(t, time):
                raise ValueError(f"Expected time object, got {type(t)}")
                
            # Ensure minutes is a valid number
            try:
                minutes = int(minutes)
            except (TypeError, ValueError):
                minutes = 0
                
            # Add minutes safely
            dt = datetime.combine(date.today(), t) + timedelta(minutes=minutes)
            result = dt.time()
            
            # Validate the result
            if not (0 <= result.hour <= 23 and 0 <= result.minute <= 59):
                raise ValueError(f"Invalid time result: {result}")
                
            return result
            
        except Exception as e:
            print(f"ERROR in _add_minutes: {e} (input: {t}, minutes: {minutes})")
            # Return a safe default time (12:00 PM) instead of the original time
            return time(12, 0)

# Example usage
if __name__ == "__main__":
    import asyncio
    from datetime import date
    from pydantic import BaseModel, Field
    from typing import List, Dict, Any, Optional
    
    # Define the necessary models for testing
    class PointOfInterest(BaseModel):
        name: str
        category: str
        duration_minutes: int
        location: str
        tags: List[str]
        description: str
        cost: Optional[float] = 0.0
        
    class TravelPlanRequest(BaseModel):
        destination: str
        duration_days: int
        travel_style: List[str]
        budget: str
        constraints: List[str]
        origin: Optional[str] = None
        start_date: Optional[date] = None
        
    class Activity(BaseModel):
        name: str
        start_time: str
        end_time: str
        location: str
        description: str
        
        def dict(self, **kwargs):
            return {
                'name': self.name,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'location': self.location,
                'description': self.description
            }
    
    class DailyItinerary(BaseModel):
        day: int
        date: str = ""
        activities: List[Dict[str, Any]] = []
        
    class TravelItinerary(BaseModel):
        destination: str
        duration_days: int
        daily_plans: List[DailyItinerary]
    
async def test_itinerary_agent(destination: str = "Paris, France", duration_days: int = 2, origin: str = "New York, USA"):
    # Create test POIs based on destination
    if "araku" in destination.lower():
        test_pois = [
            {
                "name": "Araku Valley View Point",
                "category": "Scenic Viewpoint",
                "duration_minutes": 90,
                "location": "Araku Valley, Andhra Pradesh",
                "tags": ["scenic", "nature", "photography"],
                "description": "Breathtaking views of the Araku Valley. Best visited in the morning for clear views of the valley.",
                "cost": 0.0
            },
            {
                "name": "Borra Caves",
                "category": "Cave",
                "duration_minutes": 120,
                "location": "Borra Caves Road, Borra, Andhra Pradesh",
                "tags": ["geological", "adventure", "family-friendly"],
                "description": "Million-year-old limestone caves with stunning stalactite and stalagmite formations. Well-lit pathways make it accessible.",
                "cost": 60.0
            },
            {
                "name": "Padmapuram Gardens",
                "category": "Garden",
                "duration_minutes": 60,
                "location": "Araku Valley, Andhra Pradesh",
                "tags": ["botanical", "relaxing", "family-friendly"],
                "description": "Beautiful garden with tree-top huts and a toy train. Perfect for a peaceful stroll.",
                "cost": 20.0
            },
            {
                "name": "Tribal Museum",
                "category": "Museum",
                "duration_minutes": 60,
                "location": "Araku Valley, Andhra Pradesh",
                "tags": ["cultural", "educational", "indoor"],
                "description": "Showcases the rich tribal culture and heritage of the Araku Valley region.",
                "cost": 30.0
            }
        ]
    else:
        # Default Paris POIs
        test_pois = [
            {
                "name": "Eiffel Tower",
                "category": "Landmark",
                "duration_minutes": 120,
                "location": "Champ de Mars, 5 Avenue Anatole France, 75007 Paris",
                "tags": ["iconic", "romantic", "view"],
                "description": "Iconic iron tower offering panoramic views of Paris. Book tickets in advance to skip the line.",
                "cost": 25.50
            },
            {
                "name": "Louvre Museum",
                "category": "Museum",
                "duration_minutes": 180,
                "location": "Rue de Rivoli, 75001 Paris",
                "tags": ["art", "culture", "history"],
                "description": "World's largest art museum, home to the Mona Lisa. Closed on Tuesdays.",
                "cost": 17.00
            },
            {
                "name": "Montmartre",
                "category": "Neighborhood",
                "duration_minutes": 150,
                "location": "18th arrondissement, Paris",
                "tags": ["romantic", "artsy", "views"],
                "description": "Historic district with charming streets and the Sacré-Cœur Basilica. Great for an evening stroll.",
                "cost": 0.0
            },
            {
                "name": "Seine River Cruise",
                "category": "Activity",
                "duration_minutes": 60,
                "location": "Various docks along the Seine",
                "tags": ["romantic", "scenic", "evening"],
                "description": "Evening cruise with beautiful views of Paris landmarks. Best at sunset.",
                "cost": 15.00
            }
        ]
    
    # Create a test travel request
    travel_request = TravelPlanRequest(
        destination=destination,
        duration_days=duration_days,
        travel_style=["cultural", "nature"],
        budget="mid-range",
        constraints=[],
        origin=origin,
        start_date=date.today()
    )
    
    # Initialize the agent
    agent = ItineraryAgent()
    
    # Generate the itinerary
    print("\n=== Generating Itinerary ===")
    itinerary = await agent.process(travel_request, test_pois)
    
    # Print the itinerary
    print(f"\n{itinerary.destination} Itinerary ({itinerary.duration_days} days)")
    print("=" * 50)
    
    for day_plan in itinerary.daily_plans:
        print(f"\nDay {day_plan.day}")
        print("-" * 10)
        for activity in day_plan.activities:
            print(f"{activity['start_time']} - {activity['end_time']}: {activity['name']}")
            print(f"  Location: {activity['location']}")
            print(f"  {activity['description']}")
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_itinerary_agent())
