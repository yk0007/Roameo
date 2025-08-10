"""Formatter Agent for converting travel itineraries to different formats."""
from typing import List, Dict, Any, Optional, Literal, Union
from datetime import datetime
import json

from .base import DailyItinerary, TravelItinerary, PointOfInterest

class FormatterAgent:
    """Agent responsible for formatting travel itineraries into different output formats."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.3):
        """Initialize the FormatterAgent.
        
        Args:
            model_name: Name of the LLM model (unused, for API compatibility)
            temperature: Temperature for model generation (unused, for API compatibility)
        """
        self.supported_formats = ["markdown", "json", "text", "html"]
        self.model_name = model_name
        self.temperature = temperature
    
    async def format_itinerary(
        self, 
        itinerary: Union[List[DailyItinerary], TravelItinerary],
        output_format: Literal["markdown", "json", "text", "html"] = "markdown",
        include_pois: bool = True,
        include_notes: bool = True
    ) -> str:
        """Format the itinerary into the specified output format.
        
        Args:
            itinerary: List of DailyItinerary objects or a TravelItinerary object
            output_format: Desired output format (markdown, json, text, html)
            include_pois: Whether to include point of interest details
            include_notes: Whether to include activity notes
            
        Returns:
            Formatted itinerary as a string
        """
        print(f"DEBUG: Formatting itinerary with format: {output_format}")
        print(f"DEBUG: Itinerary type: {type(itinerary)}")
        
        # Handle case where itinerary is None or empty
        if not itinerary:
            print("WARNING: Empty or None itinerary provided to formatter")
            return "## 🎯 No itinerary data available. Let's start planning your adventure!"
        
        # Handle case where we have a TravelItinerary object
        if hasattr(itinerary, 'daily_plans'):
            print(f"DEBUG: Processing TravelItinerary with {len(itinerary.daily_plans)} daily plans")
            daily_plans = itinerary.daily_plans
        # Handle case where we already have a list of DailyItinerary
        elif isinstance(itinerary, list):
            print(f"DEBUG: Processing list of {len(itinerary)} daily plans")
            daily_plans = itinerary
        else:
            print(f"WARNING: Unexpected itinerary type: {type(itinerary)}")
            return "## 🎯 Unable to process itinerary. Please try again."
            
        # If no daily plans, return a friendly message
        if not daily_plans:
            print("WARNING: No daily plans found in itinerary")
            return "## 🎯 No daily plans found in the itinerary. Let's try again!"
            
        # Get the format method
        if output_format not in self.supported_formats:
            print(f"WARNING: Unsupported format requested: {output_format}")
            output_format = "markdown"  # Fall back to markdown
            
        format_method = getattr(self, f"_format_as_{output_format}", None)
        if not format_method:
            print(f"WARNING: No formatter found for format: {output_format}")
            return "## 🎯 Unable to format itinerary. Please try a different format."
        
        try:
            # Call the appropriate format method
            print(f"DEBUG: Calling formatter for {output_format}")
            formatted = await format_method(daily_plans, include_pois, include_notes)
            
            # Ensure we have valid output
            if not formatted or not isinstance(formatted, str):
                print(f"WARNING: Formatter returned invalid output: {formatted}")
                return "## 🎯 Unable to format itinerary. Please try again."
                
            print(f"DEBUG: Successfully formatted itinerary ({len(formatted)} characters)")
            return formatted
            
        except Exception as e:
            print(f"ERROR in format_itinerary: {str(e)}")
            # Fall back to a simple text format if the main formatter fails
            try:
                return self._format_as_text_fallback(daily_plans)
            except Exception as fallback_error:
                print(f"ERROR in fallback formatter: {str(fallback_error)}")
                return "## 🎯 We encountered an error formatting your itinerary. Please try again."
    
    def _format_as_text_fallback(self, daily_plans):
        """Fallback text formatter that creates a simple text version of the itinerary."""
        output = ["=== YOUR TRAVEL ITINERARY ===\n"]
        
        # Initialize destination with a default value
        destination = "your destination"
        
        # Try to get destination from the first day's activities if available
        if daily_plans and hasattr(daily_plans[0], 'activities') and daily_plans[0].activities:
            first_activity = daily_plans[0].activities[0]
            if hasattr(first_activity, 'location') and first_activity.location:
                destination = first_activity.location
        
        for day_plan in daily_plans:
            # Get day number and date
            day_num = self._get_activity_field(day_plan, 'day', '1')
            day_date = self._get_activity_field(day_plan, 'date', '')
            
            # Add day header
            day_header = f"\n## Day {day_num}"
            if day_date:
                day_header += f" - {day_date}"
            output.append(day_header + "\n")
            
            # Add activities
            activities = self._get_activity_field(day_plan, 'activities', [])
            if not activities:
                output.append("No activities planned for this day.\n")
                continue
                
            for activity in activities:
                # Get activity details with fallbacks
                name = self._get_activity_field(activity, 'name', 'Unnamed Activity')
                start = self._get_activity_field(activity, 'start_time', '')
                end = self._get_activity_field(activity, 'end_time', '')
                location = self._get_activity_field(activity, 'location', '')
                description = self._get_activity_field(activity, 'description', '')
                
                # Format the activity
                activity_str = f"- {name}"
                if start and end:
                    activity_str += f" ({start} - {end})"
                if location:
                    activity_str += f" at {location}"
                if description:
                    activity_str += f": {description}"
                    
                output.append(activity_str + "\n")
        
        # Add a note about the return trip
        output.append("\n🔙 Remember to plan your return trip! Ensure you have transportation back to your origin.\n")
        
        return "\n".join(output)
    
    def _get_activity_field(self, activity, field_name, default=None):
        """Safely get a field from an activity, whether it's a dict or object."""
        if activity is None:
            return default
            
        if hasattr(activity, 'get') and callable(activity.get):
            # Handle dictionary access
            return activity.get(field_name, default)
        # Handle object attribute access
        return getattr(activity, field_name, default)
    
    async def _format_as_markdown(
        self, 
        itinerary: Union[List[DailyItinerary], TravelItinerary],
        include_pois: bool = True,
        include_notes: bool = True
    ) -> str:
        """Format the itinerary as an engaging Markdown document with rich descriptions."""
        # Initialize with default values
        daily_plans = []
        destination = 'Your Destination'
        duration_days = 1
        
        # Handle both List[DailyItinerary] and TravelItinerary inputs
        if hasattr(itinerary, 'daily_plans'):  # It's a TravelItinerary object
            daily_plans = getattr(itinerary, 'daily_plans', [])
            destination = getattr(itinerary, 'destination', 'Your Destination')
            duration_days = getattr(itinerary, 'duration_days', len(daily_plans))
        elif isinstance(itinerary, list):  # It's already a list of daily plans
            daily_plans = itinerary
        
        # Add a beautiful header with destination and trip duration
        output = [
            f"# 🌍 {destination.upper()} ADVENTURE 🌎\n\n"
            f"*{duration_days} days of unforgettable experiences*\n\n"
            "---\n"
        ]
        
        if not daily_plans:
            return "## 🎯 No itinerary data available. Let's start planning your adventure!"
        
        # Add a trip overview section
        output.append("## 🌟 Trip Overview\n")
        output.append("Get ready for an amazing journey filled with incredible experiences. "
                     "Here's what your adventure looks like!\n")
        
        # Add a quick day-by-day preview
        output.append("### 📅 Quick Glance\n")
        for i, day_plan in enumerate(daily_plans, 1):
            day_activities = self._get_activity_field(day_plan, 'activities', [])
            highlight = self._get_activity_field(day_activities[0], 'name', 'Exciting activities') if day_activities else 'Free exploration'
            output.append(f"- **Day {i}:** {highlight}")
        output.append("\n---\n")
            
        for day_plan in daily_plans:
            # Add day header with emoji
            day_number = self._get_activity_field(day_plan, 'day', 1)
            day_date = self._get_activity_field(day_plan, 'date')
            date_str = day_date.strftime("%A, %B %d, %Y") if day_date else f"Day {day_number}"
            
            # Add a beautiful day header with a divider
            output.append(f"## 🌄 {date_str.upper()} 🌇\n")
            
            # Add day introduction if available
            day_intro = self._get_activity_field(day_plan, 'introduction', 
                                               f"Day {day_number} of your {destination} adventure!")
            output.append(f"*{day_intro}*\n")
            
            # Get activities, handling both dict and object access
            activities = self._get_activity_field(day_plan, 'activities', [])
            
            if not activities:
                output.append("*A free day to explore at your own pace!*\n")
                output.append("💡 **Tip:** This is a great opportunity to revisit your favorite spots or discover hidden gems!\n")
                output.append("---\n")
                continue
                
            # Add activities with rich formatting
            output.append("### 🗓️ Today's Schedule\n")
            
            for activity in activities:
                if not activity:
                    continue
                    
                name = self._get_activity_field(activity, 'name', 'Unnamed Activity')
                start_time = self._get_activity_field(activity, 'start_time', '09:00')
                end_time = self._get_activity_field(activity, 'end_time', '10:00')
                description = self._get_activity_field(activity, 'description', 
                                                     'An exciting experience awaits!')
                location = self._get_activity_field(activity, 'location', 'Various locations')
                activity_type = self._get_activity_field(activity, 'type', 'activity').lower()
                
                # Select appropriate emoji based on activity type
                emoji_map = {
                    'breakfast': '🍳', 'lunch': '🍽️', 'dinner': '🍲', 'meal': '🍴',
                    'sightseeing': '🏛️', 'museum': '🏛️', 'landmark': '🗼', 'beach': '🏖️',
                    'hiking': '🥾', 'nature': '🌲', 'park': '🌳', 'shopping': '🛍️',
                    'adventure': '🏄', 'water': '🌊', 'mountain': '⛰️', 'city': '🏙️',
                    'tour': '🚶', 'walking': '🚶', 'food': '🍜', 'drink': '🍹',
                    'coffee': '☕', 'photo': '📸', 'relax': '😌', 'spa': '💆',
                    'transport': '🚗', 'train': '🚆', 'bus': '🚌', 'flight': '✈️',
                    'hotel': '🏨', 'checkin': '🏠', 'checkout': '🏠', 'free': '✨'
                }
                
                # Default emoji
                emoji = '📍'  # Default marker
                for key, e in emoji_map.items():
                    if key in activity_type.lower():
                        emoji = e
                        break
                
                # Format the activity with rich details
                time_str = f"**{self._format_time(start_time)} - {self._format_time(end_time)}**"
                
                # Create the activity header with emoji and name
                activity_line = f"### {emoji} {name}\n"
                
                # Add time and location
                activity_line += f"⏰ **When:** {time_str}\n"
                if location and location.lower() not in ['various', 'various locations', 'tbd']:
                    activity_line += f"📍 **Where:** {location}\n"
                
                # Add description with enhanced formatting
                if description:
                    activity_line += f"\n{self._enhance_description(description)}\n"
                
                # Add duration if available
                duration = self._get_activity_field(activity, 'duration')
                if duration:
                    activity_line += f"\n⏳ **Duration:** ~{duration} minutes\n"
                
                # Add cost estimate if available
                cost = self._get_activity_field(activity, 'cost')
                if cost:
                    activity_line += f"💰 **Estimated Cost:** ₹{cost}\n"
                
                # Add any special notes or tips
                if include_notes:
                    notes = self._get_activity_field(activity, 'notes')
                    if notes:
                        activity_line += f"\n💡 **Local Tip:** {notes}\n"
                
                # Add a subtle divider
                activity_line += "\n---\n"
                
                output.append(activity_line)
        
        # Add a conclusion
        output.append("\n## 🎉 Trip Complete!")
        output.append("\nWhat an amazing adventure! We hope you've had a wonderful time exploring. "
                    "Safe travels on your journey home!")
        
        # Add a final divider
        output.append("\n---\n")
        output.append("*Itinerary generated by Agentic Travel Planner* 🚀")
        
        # Join all lines into a single string and ensure we return a string
        formatted_output = "\n".join(output)
        if not isinstance(formatted_output, str):
            return "## 🎯 Error formatting itinerary. Please try again."
        return formatted_output
    
    def _enhance_description(self, description: str) -> str:
        """Enhance the activity description with markdown formatting."""
        # Add emphasis to key phrases
        enhancements = {
            'recommended': '🌟 **Recommended**:',
            'must-see': '✨ **Must-see**:',
            'hidden gem': '💎 **Hidden gem**:',
            'photo spot': '📸 **Great photo spot**:',
            'local favorite': '❤️ **Local favorite**:',
            'best time': '⏱️ **Best time**:',
            'budget': '💵 **Budget**:',
            'pro tip': '💡 **Pro tip**:',
            'insider tip': '🤫 **Insider tip**:',
            'fun fact': '📚 **Fun fact**:'
        }
        
        # Apply enhancements
        for key, replacement in enhancements.items():
            if key in description.lower():
                # Find the position and replace with markdown
                pos = description.lower().find(key)
                if pos >= 0:
                    # Preserve the original capitalization
                    original_phrase = description[pos:pos+len(key)]
                    description = description.replace(original_phrase, replacement, 1)
        
        return description
    
    async def _format_as_text(
        self, 
        itinerary: List[DailyItinerary],
        include_pois: bool = True,
        include_notes: bool = True
    ) -> str:
        """Format the itinerary as plain text."""
        output = ["TRAVEL ITINERARY\n" + "="*40]
        
        for day in itinerary:
            # Add day header
            day_number = self._get_activity_field(day, 'day', 1)
            day_date = self._get_activity_field(day, 'date')
            date_str = day_date.strftime("%A, %B %d, %Y") if day_date else f"Day {day_number}"
            output.append(f"\n{date_str}")
            output.append("-" * len(date_str))
            
            # Get activities, handling both dict and object access
            activities = self._get_activity_field(day, 'activities', [])
            
            # Add activities
            for activity in activities:
                name = self._get_activity_field(activity, 'name', 'Activity')
                start_time = self._get_activity_field(activity, 'start_time', '09:00')
                end_time = self._get_activity_field(activity, 'end_time', '10:00')
                description = self._get_activity_field(activity, 'description', '')
                location = self._get_activity_field(activity, 'location', '')
                
                time_str = f"{self._format_time(start_time)} - {self._format_time(end_time)}"
                output.append(f"\n{time_str}: {name}")
                
                if include_notes and activity.description:
                    output.append(f"  {activity.description}")
                
                if include_pois and hasattr(activity, 'location') and activity.location:
                    output.append(f"  Location: {activity.location}")
            
            output.append("")  # Add space between days
        
        return "\n".join(output).strip()
    
    async def _format_as_json(
        self, 
        itinerary: List[DailyItinerary],
        include_pois: bool = True,
        include_notes: bool = True
    ) -> str:
        """Format the itinerary as JSON."""
        result = []
        
        for day in itinerary:
            day_number = self._get_activity_field(day, 'day', 1)
            day_date = self._get_activity_field(day, 'date')
            activities = self._get_activity_field(day, 'activities', [])
            
            day_data = {
                "day": day_number,
                "date": day_date.isoformat() if day_date else None,
                "activities": []
            }
            
            for activity in activities:
                name = self._get_activity_field(activity, 'name', 'Activity')
                start_time = self._get_activity_field(activity, 'start_time', '09:00')
                end_time = self._get_activity_field(activity, 'end_time', '10:00')
                description = self._get_activity_field(activity, 'description', '')
                location = self._get_activity_field(activity, 'location', '')
                
                activity_data = {
                    "name": name,
                    "start_time": self._format_time(start_time),
                    "end_time": self._format_time(end_time),
                    "location": location if location else None,
                    "description": description if description else None
                }
                
                if not include_pois:
                    activity_data.pop("location", None)
                if not include_notes:
                    activity_data.pop("description", None)
                    
                day_data["activities"].append(activity_data)
                
            result.append(day_data)
            
        return json.dumps(result, indent=2)
    
    async def _format_as_html(
        self, 
        itinerary: List[DailyItinerary],
        include_pois: bool = True,
        include_notes: bool = True
    ) -> str:
        """Format the itinerary as HTML."""
        css = """
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            h2 { color: #2980b9; margin-top: 30px; }
            .day { margin-bottom: 30px; }
            .activity { margin-bottom: 15px; }
            .time { font-weight: bold; color: #e74c3c; }
            .location { color: #7f8c8d; font-style: italic; }
            .description { margin-top: 5px; color: #34495e; }
        </style>
        """
        
        html = [f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Travel Itinerary</title>
            {css}
        </head>
        <body>
            <h1>Travel Itinerary</h1>
        """.format(css=css)]
        
        for day in itinerary:
            day_number = self._get_activity_field(day, 'day', 1)
            day_date = self._get_activity_field(day, 'date')
            activities = self._get_activity_field(day, 'activities', [])
            
            date_str = day_date.strftime("%A, %B %d, %Y") if day_date else f"Day {day_number}"
            html.append(f'<div class="day">')
            html.append(f'<h2>{date_str}</h2>')
            
            for activity in activities:
                name = self._get_activity_field(activity, 'name', 'Activity')
                start_time = self._get_activity_field(activity, 'start_time', '09:00')
                end_time = self._get_activity_field(activity, 'end_time', '10:00')
                description = self._get_activity_field(activity, 'description', '')
                location = self._get_activity_field(activity, 'location', '')
                
                time_str = f"<span class='time'>{self._format_time(start_time)} - {self._format_time(end_time)}</span>"
                html.append(f'<div class="activity">')
                html.append(f'<div>{time_str}: {name}</div>')
                
                if include_notes and description:
                    html.append(f'<div class="description">{description}</div>')
                    
                if include_pois and location:
                    html.append(f'<div class="location">📍 {location}</div>')
                    
                html.append('</div>')
            
            html.append('</div>')
        
        html.append("""
            </body>
            </html>
        """)
        
        return '\n'.join(html)
    
    def _format_time(self, time_value) -> str:
        """Format a time value into a consistent format.
        
        Args:
            time_value: Can be a datetime.time object, a string in HH:MM format, or None
            
        Returns:
            Formatted time string in HH:MM format
        """
        if not time_value:
            return ""
            
        # If it's already a time object, format it
        if hasattr(time_value, 'strftime'):
            return time_value.strftime("%H:%M")
            
        # If it's already a string, try to parse it
        if isinstance(time_value, str):
            # Handle ISO format strings that might include timezone
            time_str = time_value.split('T')[-1].split('+')[0]
            
            # Try different time formats
            time_formats = ["%H:%M", "%H:%M:%S", "%I:%M %p", "%I:%M%p"]
            
            for fmt in time_formats:
                try:
                    time_obj = datetime.strptime(time_str.strip(), fmt)
                    return time_obj.strftime("%H:%M")
                except ValueError:
                    continue
        
        # If all else fails, return the string representation
        return str(time_value)
