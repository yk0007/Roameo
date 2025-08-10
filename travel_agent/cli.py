"""Command-line interface for the Agentic Travel Planner."""
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from .workflow import travel_planner_workflow

# ANSI color codes for console output
COLORS = {
    'HEADER': '\033[95m',
    'BLUE': '\033[94m',
    'CYAN': '\033[96m',
    'GREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m',
    'BOLD': '\033[1m',
    'UNDERLINE': '\033[4m'
}

def print_header():
    """Print the application header."""
    print(f"{COLORS['HEADER']}{COLORS['BOLD']}")
    print("=" * 60)
    print("AGENTIC TRAVEL PLANNER".center(60))
    print("=" * 60)
    print(f"{COLORS['ENDC']}")

def print_section(title: str):
    """Print a section header."""
    print(f"\n{COLORS['BLUE']}{COLORS['BOLD']}=== {title.upper()} ==={COLORS['ENDC']}")

def print_success(message: str):
    """Print a success message."""
    print(f"{COLORS['GREEN']}✓ {message}{COLORS['ENDC']}")

def print_warning(message: str):
    """Print a warning message."""
    print(f"{COLORS['WARNING']}⚠️  {message}{COLORS['ENDC']}")

def print_error(message: str):
    """Print an error message."""
    print(f"{COLORS['FAIL']}✗ {message}{COLORS['ENDC']}")

def print_info(message: str):
    """Print an info message."""
    print(f"{COLORS['CYAN']}ℹ️  {message}{COLORS['ENDC']}")

def format_duration(minutes: int) -> str:
    """Format duration in minutes to a human-readable string."""
    if minutes < 60:
        return f"{minutes} min"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}min"

async def get_output_format() -> str:
    """Prompt user to select an output format."""
    print("\n" + "-" * 60)
    print("Select output format:")
    print("1. Markdown (default)")
    print("2. JSON")
    print("3. Plain Text")
    print("4. HTML")
    
    format_map = {
        "1": "markdown",
        "2": "json",
        "3": "text",
        "4": "html"
    }
    
    choice = input("\nEnter your choice (1-4, default is 1): ").strip()
    return format_map.get(choice, "markdown")

async def run_travel_planner():
    """Run the travel planner CLI."""
    print_header()
    
    # Get user input
    print("\n" + "-" * 60)
    print("Please describe your trip in natural language (e.g., 'I want to visit Paris for 5 days in June'):")
    user_input = input("\n> ").strip()
    
    if not user_input:
        print_error("No input provided. Exiting...")
        return
        
    # Get output format
    output_format = await get_output_format()
    
    # Import the AgentState class
    from .workflow import AgentState
    
    # Initialize the state using the Pydantic model
    state = AgentState(
        user_input=user_input,
        output_format=output_format,
        travel_request=None,
        suggested_pois=[],
        selected_pois=[],
        itinerary=[],
        formatted_itinerary="",
        messages=[],
        status="planning",
        error=None
    )
    
    # Run the workflow
    print_section("Planning Your Trip")
    print("Processing your request...\n")
    
    try:
        result = await travel_planner_workflow.ainvoke(state)
        
        # Display results
        if result.get('error'):
            print_error(f"Error: {result['error']}")
            return
        
        # Display the formatted output if available
        if 'formatter_output' in result and result['formatter_output']:
            print_section("Your Travel Itinerary")
            print()  # Add a blank line before the formatted output
            
            # For JSON output, pretty print it
            if output_format == "json":
                try:
                    if isinstance(result['formatter_output'], str):
                        parsed_json = json.loads(result['formatter_output'])
                        print(json.dumps(parsed_json, indent=2, ensure_ascii=False))
                    else:
                        print(json.dumps(result['formatter_output'], indent=2, ensure_ascii=False))
                except (json.JSONDecodeError, TypeError):
                    print(result['formatter_output'])
            else:
                print(result['formatter_output'])
        
        # Fallback to basic display if no formatted output but we have an itinerary
        elif 'itinerary_output' in result and result['itinerary_output']:
            print_section("Your Travel Itinerary (Basic)")
            itinerary = result['itinerary_output']
            
            # Handle both list of DailyItinerary and TravelItinerary objects
            daily_plans = getattr(itinerary, 'daily_plans', None)
            if daily_plans is None and hasattr(itinerary, '__iter__'):
                daily_plans = itinerary
            
            if daily_plans:
                for day in daily_plans:
                    day_num = getattr(day, 'day', getattr(day, 'day_number', 'Unknown'))
                    print(f"\n{COLORS['BOLD']}Day {day_num}{COLORS['ENDC']}")
                    print("-" * (4 + len(str(day_num))))
                    
                    activities = getattr(day, 'activities', [])
                    for activity in activities:
                        # Handle both Activity objects and dictionaries
                        if hasattr(activity, 'start_time'):
                            start_time = activity.start_time
                            end_time = getattr(activity, 'end_time', '')
                            name = getattr(activity, 'name', 'Unnamed Activity')
                            location = getattr(activity, 'location', '')
                            description = getattr(activity, 'description', '')
                        elif isinstance(activity, dict):
                            start_time = activity.get('start_time', '')
                            end_time = activity.get('end_time', '')
                            name = activity.get('name', 'Unnamed Activity')
                            location = activity.get('location', '')
                            description = activity.get('description', '')
                        else:
                            continue
                        
                        # Format times if they're time objects
                        if hasattr(start_time, 'strftime'):
                            start_time = start_time.strftime('%I:%M %p')
                        if hasattr(end_time, 'strftime'):
                            end_time = end_time.strftime('%I:%M %p')
                            
                        print(f"\n{start_time} - {end_time}: {name}")
                        if location:
                            print(f"  📍 {location}")
                        if description:
                            print(f"  {description}")
        
        # Display any messages
        if 'messages' in result and result['messages']:
            print_section("Messages")
            for msg in result['messages']:
                if hasattr(msg, 'content'):
                    print(f"- {msg.content}")
                else:
                    print(f"- {msg}")
        
        print_success("\nYour travel plan is ready! Have a great trip! 🎉")
        
    except Exception as e:
        print_error(f"An unexpected error occurred: {str(e)}")
        if 'result' in locals():
            print("\nDebug information:")
            print(json.dumps(result, indent=2, default=str))

def main():
    """Main entry point for the CLI."""
    try:
        asyncio.run(run_travel_planner())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        print_error(f"An error occurred: {str(e)}")
    finally:
        print("\nThank you for using the Agentic Travel Planner!")

if __name__ == "__main__":
    main()
