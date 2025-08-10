"""Calendar Agent for managing and visualizing travel plans."""
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, date, time, timedelta
import json
from pathlib import Path
import webbrowser
import tempfile

from jinja2 import Environment, FileSystemLoader

from .models import TravelItinerary, Activity, DailyItinerary
from .base import BaseAgent

class CalendarAgent(BaseAgent):
    """Agent responsible for managing and visualizing travel plans in a calendar format."""
    
    def __init__(self, model_name: str = None, temperature: float = 0.3, templates_dir: str = "templates"):
        """
        Initialize the CalendarAgent.
        
        Args:
            model_name: Name of the LLM model (unused, for API compatibility)
            temperature: Temperature for model generation (unused, for API compatibility)
            templates_dir: Directory containing HTML templates
        """
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.temperature = temperature
        
        # Initialize Jinja2 environment
        self.env = Environment(loader=FileSystemLoader(str(self.templates_dir)))
        
        # Create default template if it doesn't exist
        self._ensure_default_template()
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the calendar generation request.
        
        Args:
            input_data: Dictionary containing:
                - itinerary: TravelItinerary object or list of DailyItinerary
                - output_format: Desired output format (e.g., 'html', 'json')
                - open_in_browser: Whether to open the result in a web browser (for HTML)
                
        Returns:
            Dictionary containing the calendar data or path to the generated file
        """
        try:
            itinerary = input_data.get("itinerary")
            output_format = input_data.get("output_format", "html")
            open_in_browser = input_data.get("open_in_browser", True)
            
            if not itinerary:
                return {
                    "status": "error",
                    "error": "No itinerary provided for calendar generation"
                }
            
            # Handle different input types
            if isinstance(itinerary, list):
                # Convert list of DailyItinerary to TravelItinerary
                if not all(isinstance(day, (DailyItinerary, dict)) for day in itinerary):
                    return {
                        "status": "error",
                        "error": "Invalid itinerary format. Expected List[DailyItinerary] or TravelItinerary"
                    }
                
                # Create a temporary TravelItinerary if we have a list
                temp_itinerary = TravelItinerary(
                    destination="Multiple Destinations",
                    duration_days=len(itinerary),
                    daily_plans=itinerary
                )
                itinerary = temp_itinerary
            
            # Generate the calendar based on the requested format
            if output_format.lower() == "html":
                result = await self.generate_html_calendar(itinerary, open_in_browser)
            elif output_format.lower() == "json":
                result = await self.generate_json_calendar(itinerary)
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported output format: {output_format}. Supported formats: html, json"
                }
            
            return {
                "status": "success",
                "format": output_format,
                "data": result,
                "metadata": {
                    "model": self.model_name,
                    "timestamp": str(datetime.now().isoformat())
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to generate calendar: {str(e)}",
                "metadata": {
                    "model": self.model_name,
                    "timestamp": str(datetime.now().isoformat())
                }
            }
    
    def _ensure_default_template(self) -> None:
        """Ensure the default calendar template exists."""
        template_path = self.templates_dir / "calendar_template.html"
        if not template_path.exists():
            template_path.write_text("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Travel Itinerary: {{ itinerary.destination }}</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid #eee;
        }
        .header h1 {
            color: #2c3e50;
            margin-bottom: 5px;
        }
        .trip-dates {
            color: #7f8c8d;
            font-size: 1.1em;
            margin-bottom: 20px;
        }
        .calendar {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .day-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .day-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.15);
        }
        .day-header {
            background-color: #3498db;
            color: white;
            padding: 12px 15px;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .day-number {
            font-size: 1.5em;
            font-weight: bold;
        }
        .day-name {
            font-size: 1.1em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .activities {
            padding: 15px;
        }
        .activity {
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }
        .activity:last-child {
            border-bottom: none;
            margin-bottom: 0;
            padding-bottom: 0;
        }
        .time {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
            display: flex;
            justify-content: space-between;
        }
        .activity-title {
            font-weight: 600;
            color: #34495e;
            margin-bottom: 5px;
        }
        .activity-location, .activity-notes {
            font-size: 0.9em;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .activity-notes {
            font-style: italic;
            color: #95a5a6;
        }
        .no-activities {
            color: #95a5a6;
            font-style: italic;
            text-align: center;
            padding: 20px 0;
        }
        .trip-summary {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            border-left: 4px solid #3498db;
        }
        .summary-item {
            display: flex;
            margin-bottom: 8px;
        }
        .summary-label {
            font-weight: 600;
            min-width: 150px;
            color: #2c3e50;
        }
        .summary-value {
            flex: 1;
        }
        @media print {
            body {
                padding: 0;
                background: white;
            }
            .header {
                text-align: left;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }
            .calendar {
                display: block;
            }
            .day-card {
                page-break-inside: avoid;
                margin-bottom: 20px;
                box-shadow: none;
                border: 1px solid #ddd;
            }
            .no-print {
                display: none;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ itinerary.destination }} Itinerary</h1>
        <div class="trip-dates">
            {{ itinerary.start_date.strftime('%B %d, %Y') }} - {{ itinerary.end_date.strftime('%B %d, %Y') }}
            ({{ (itinerary.end_date - itinerary.start_date).days + 1 }} days)
        </div>
    </div>

    <div class="trip-summary">
        <h2>Trip Summary</h2>
        <div class="summary-item">
            <div class="summary-label">Destination:</div>
            <div class="summary-value">{{ itinerary.destination }}</div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Travel Dates:</div>
            <div class="summary-value">
                {{ itinerary.start_date.strftime('%A, %B %d, %Y') }} to {{ itinerary.end_date.strftime('%A, %B %d, %Y') }}
            </div>
        </div>
        <div class="summary-item">
            <div class="summary-label">Duration:</div>
            <div class="summary-value">
                {{ (itinerary.end_date - itinerary.start_date).days + 1 }} days
            </div>
        </div>
        {% if itinerary.travel_style %}
        <div class="summary-item">
            <div class="summary-label">Travel Style:</div>
            <div class="summary-value">
                {{ itinerary.travel_style|join(', ')|title }}
            </div>
        </div>
        {% endif %}
    </div>

    <h2>Daily Itinerary</h2>
    <div class="calendar">
        {% for day in itinerary.daily_plans %}
        <div class="day-card">
            <div class="day-header">
                <span class="day-name">Day {{ day.day }}</span>
                <span class="day-number">{{ day.date.strftime('%a, %b %d') }}</span>
            </div>
            <div class="activities">
                {% if day.activities %}
                    {% for activity in day.activities %}
                    <div class="activity">
                        <div class="time">
                            <span>{{ activity.start_time }}</span>
                            <span>{% if activity.end_time %}{{ activity.end_time }}{% endif %}</span>
                        </div>
                        <div class="activity-title">{{ activity.name }}</div>
                        {% if activity.location %}
                        <div class="activity-location">📍 {{ activity.location }}</div>
                        {% endif %}
                        {% if activity.description %}
                        <div class="activity-description">{{ activity.description }}</div>
                        {% endif %}
                        {% if activity.notes %}
                        <div class="activity-notes">📝 {{ activity.notes }}</div>
                        {% endif %}
                    </div>
                    {% endfor %}
                {% else %}
                    <div class="no-activities">No activities planned for this day.</div>
                {% endif %}
            </div>
        </div>
        {% endfor %}
    </div>

    <div class="no-print" style="margin-top: 40px; text-align: center; padding: 20px 0; color: #7f8c8d; font-size: 0.9em;">
        <p>Generated on {{ now.strftime('%B %d, %Y at %I:%M %p') }} | Agentic Travel Planner</p>
        <button onclick="window.print()" style=\"padding: 8px 16px; background-color: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; margin-top: 10px;\">
            Print Itinerary
        </button>
    </div>
</body>
</html>
""")
