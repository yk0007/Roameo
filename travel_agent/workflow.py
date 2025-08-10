"""LangGraph workflow for the Agentic Travel Planner system.

This module defines the main workflow that coordinates all the agents
in the travel planning process.
"""
import logging
from typing import Dict, List, Optional, Any, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict, Annotated

from travel_agent.models import TravelPlanRequest, PointOfInterest, TravelItinerary, BudgetLevel
from travel_agent.planner_agent import PlannerAgent
from travel_agent.explorer_agent import ExplorerAgent
from travel_agent.itinerary_agent import ItineraryAgent
from travel_agent.formatter_agent import FormatterAgent
from travel_agent.budget_agent import BudgetCalculationAgent
from travel_agent.travel_selections import TravelSelections

class AgentState(TypedDict):
    """State for the travel planning workflow with all agent outputs."""
    # Core state that changes during execution
    current_step: str
    
    # User input
    user_input: str
    
    # Agent outputs
    travel_request: Optional[TravelPlanRequest]
    explorer_output: Optional[List[PointOfInterest]]
    budget_output: Optional[Dict[str, Any]]
    itinerary_output: Optional[TravelItinerary]
    formatter_output: Optional[Dict[str, str]]
    
    # Error handling
    error: Optional[str]
    
    # Messages for user feedback
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y if y else x]
    
    # Output format
    output_format: str

def create_travel_planner_workflow(
    model_name: Optional[str] = None,  # Will use default from ModelConfig if None
    temperature: float = 0.3,
    debug: bool = False,
    default_currency: str = "INR"
) -> StateGraph:
    """Create and configure the travel planning workflow with all agents.
    
    Args:
        model_name: Name of the LLM model to use
        temperature: Temperature for model generation
        debug: Enable debug logging
        default_currency: Default currency for budget calculations (default: 'INR')
    """
    # Initialize only the essential agents for the core workflow
    # If model_name is None, each agent will use the default from ModelConfig
    planner_agent = PlannerAgent(model_name=model_name, temperature=temperature)
    explorer_agent = ExplorerAgent(model_name=model_name, temperature=temperature)
    budget_agent = BudgetCalculationAgent(model_name=model_name, temperature=temperature)
    itinerary_agent = ItineraryAgent(model_name=model_name, temperature=temperature)
    formatter_agent = FormatterAgent(model_name=model_name, temperature=temperature)
    
    # Log the current model configuration
    from travel_agent.utils.model_config import ModelConfig
    provider = ModelConfig.get_provider()
    current_model = ModelConfig.get_model_name(provider)
    logging.info(f"Using {provider.upper()} model: {current_model} with temperature={temperature}")
    
    # Initialize travel selections
    travel_selections = TravelSelections()
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
    logger = logging.getLogger(__name__)

    # Define nodes
    async def parse_user_input(state: AgentState) -> AgentState:
        """Parse the user's natural language input into a structured travel request."""
        try:
            logger.debug("Starting parse_user_input")
            if not state.get('user_input'):
                raise ValueError("No user input provided")
            
            # Process the user input to create a travel request
            travel_request = await planner_agent.process(state['user_input'])
            
            # Debug logging
            logger.debug(f"TravelPlanRequest after processing: {travel_request}")
            logger.debug(f"TravelPlanRequest fields: {travel_request.__fields__}")
            logger.debug(f"TravelPlanRequest interests: {getattr(travel_request, 'interests', 'NOT FOUND')}")
            
            # Ensure all required fields are set with defaults if missing
            if not hasattr(travel_request, 'travel_style') or not travel_request.travel_style:
                travel_request.travel_style = ["cultural"]
            if not hasattr(travel_request, 'budget') or not travel_request.budget:
                travel_request.budget = "mid-range"
            if not hasattr(travel_request, 'interests') or not travel_request.interests:
                travel_request.interests = ["sightseeing"]
            
            # Debug logging after setting defaults
            logger.debug(f"TravelPlanRequest after setting defaults: {travel_request}")
            logger.debug(f"TravelPlanRequest interests after setting defaults: {travel_request.interests}")
            
            # Create a new state with only the updated fields
            return {
                **state,
                'current_step': 'parse_user_input',
                'travel_request': travel_request,
                'messages': [
                    *state.get('messages', []),
                    AIMessage(content=f"Successfully parsed your travel request for {travel_request.destination}")
                ]
            }
            
        except Exception as e:
            error_msg = f"Error parsing user input: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                'current_step': 'parse_user_input',
                'error': error_msg
            }

    async def explore_pois(state: AgentState) -> AgentState:
        """Explore points of interest based on the travel request."""
        try:
            logger.debug("Starting explore_pois")
            request = state.get('travel_request')
            if not request:
                raise ValueError("No travel request available for POI exploration")
                
            # Create a travel request with the necessary parameters
            from travel_agent.models import TravelPlanRequest
            
            # Check if we already have a valid travel request
            if not isinstance(request, TravelPlanRequest):
                # Create a minimal travel request if needed
                request = TravelPlanRequest(
                    destination=request.destination,
                    duration_days=request.duration_days,
                    travelers=1,  # Default value
                    budget_level=getattr(request, 'budget_level', None),
                    interests=getattr(request, 'interests', [])
                )
            
            # Use the explorer agent to find POIs
            pois = await explorer_agent.process(travel_request=request)
            
            # Create a new state with only the updated fields
            return {
                **state,
                'current_step': 'explore_pois',
                'explorer_output': pois,
                'messages': [
                    *state.get('messages', []),
                    AIMessage(
                        content=f"Found {len(pois)} points of interest for your trip to {request.destination}"
                    )
                ]
            }
            
        except Exception as e:
            error_msg = f"Error exploring POIs: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                'current_step': 'explore_pois',
                'error': error_msg
            }
    
    async def plan_budget(state: AgentState) -> AgentState:
        """Plan the budget based on user preferences and selections with currency support."""
        try:
            logger.debug("Starting plan_budget")
            request = state.get('travel_request')
            if not request:
                raise ValueError("No travel request available for budget planning")
            
            # Get destination and duration
            destination = request.destination
            duration = request.duration_days
            budget_key = f"budget_shown_{destination}_{duration}"
            
            # Check if we've already processed the budget in this state
            if state.get(budget_key, False):
                logger.debug(f"Budget already shown for {destination} ({duration} days)")
                return state
            
            # Create a new state to avoid modifying the original, preserving explorer_output
            new_state = {
                **state,
                'explorer_output': state.get('explorer_output', []),  # Preserve existing POIs
                'current_step': 'plan_budget'
            }
            
            # Check if we already have a budget in the messages
            budget_found = False
            filtered_messages = []
            
            for msg in state.get('messages', []):
                if isinstance(msg, AIMessage) and "Budget Summary for" in msg.content:
                    # If this is the first budget message for this destination/duration, keep it
                    if not budget_found and f"Budget Summary for {destination} ({duration} days)" in msg.content:
                        filtered_messages.append(msg)
                        budget_found = True
                    # Skip all other budget messages
                    continue
                filtered_messages.append(msg)
            
            # If we found a budget message, mark it as shown and return
            if budget_found:
                logger.debug(f"Found existing budget for {destination} ({duration} days)")
                new_state[budget_key] = True
                new_state['messages'] = filtered_messages
                return new_state
            
            # If we get here, we need to calculate a new budget
            logger.debug(f"Calculating new budget for {destination} ({duration} days)")
            
            # Mark that we're showing this budget
            new_state[budget_key] = True
            new_state['messages'] = filtered_messages
            
            # Get budget level from request or use default
            budget_level = getattr(request, 'budget_level', BudgetLevel.MID_RANGE)
            
            # Get target currency from request or use default
            target_currency = getattr(request, 'currency', default_currency).upper()
            
            # Calculate budget with currency conversion
            budget = await budget_agent.estimate_budget(
                destination=destination,
                duration_days=duration,
                budget_level=budget_level,
                travel_style=request.interests,
                group_size=request.group_size,
                additional_notes=getattr(request, 'additional_notes', '') or "",
                target_currency=target_currency
            )
            
            # Get budget breakdown details
            currency = budget.get('currency', target_currency).upper()
            total_cost = float(budget.get('total_estimated_cost', 0))
            breakdown = budget.get('budget_breakdown', {})
            
            # Format the budget message with detailed breakdown
            budget_message = [
                f"## 💰 Budget Summary for {destination} ({duration} days)",
                f"**Total Estimated Cost:** {budget_agent.format_currency(total_cost, currency)}",
                "",
                "### 📊 Cost Breakdown:"
            ]
            
            # Add each category's details
            for category, details in breakdown.items():
                if isinstance(details, dict) and 'total_estimate' in details:
                    daily = details.get('daily_estimate', 0)
                    total = details.get('total_estimate', 0)
                    notes = details.get('notes', '')
                    
                    budget_message.append(
                        f"- **{category.title()}**: {budget_agent.format_currency(total, currency)} "
                        f"({budget_agent.format_currency(daily, currency)} per day) {notes}"
                    )
            
            # Add budget level information
            budget_level = getattr(request, 'budget_level', 'mid-range')
            budget_message.extend([
                "",
                f"*Budget Level: {budget_level.value if hasattr(budget_level, 'value') else budget_level}*"
            ])
            
            # Add filtered recommendations (remove any about package tours and duplicates)
            if 'recommendations' in budget and budget['recommendations']:
                # Filter out package tour recommendations and remove duplicates
                seen_tips = set()
                filtered_tips = []
                for tip in budget['recommendations']:
                    tip_lower = tip.lower()
                    if 'package tour' not in tip_lower and tip_lower not in seen_tips:
                        filtered_tips.append(tip)
                        seen_tips.add(tip_lower)
                
                # Only add if we have tips after filtering
                if filtered_tips:
                    budget_message.extend(["", "### 💡 Money-Saving Tips:"])
                    budget_message.extend(f"- {tip}" for tip in filtered_tips)
            
            # Combine the budget message
            budget_content = "\n".join(budget_message)
            
            # Add the new budget message to the state
            new_state['current_step'] = 'plan_budget'
            new_state['budget_output'] = budget
            
            # Check if we've already added a budget for this destination and duration
            budget_key = f"budget_{destination}_{duration}"
            
            # Only add the budget message if it's not already in the state
            if budget_key not in state.get('_budget_keys', set()):
                # Add the budget content to messages
                filtered_messages.append(AIMessage(content=budget_content))
                
                # Update the budget keys in the state
                budget_keys = set(state.get('_budget_keys', set()))
                budget_keys.add(budget_key)
                
                # Update the state with the new messages and budget keys
                new_state = {
                    **state,
                    'messages': filtered_messages,
                    '_budget_keys': budget_keys,
                    'current_step': 'plan_budget',
                    'budget_output': budget  # Use the correct variable name 'budget' instead of 'budget_output'
                }
                
                logger.debug(f"Added new budget for {destination} ({duration} days)")
            else:
                logger.debug(f"Budget for {destination} ({duration} days) already exists, skipping")
                new_state = {
                    **state,
                    'current_step': 'plan_budget',
                    'budget_output': budget  # Use the correct variable name 'budget' instead of 'budget_output'
                }
                
            return new_state
            
        except Exception as e:
            error_msg = f"Error in budget planning: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                'current_step': 'plan_budget',
                'error': error_msg
            }
    
    async def create_itinerary(state: AgentState) -> AgentState:
        """Create a travel itinerary based on selected POIs."""
        try:
            logger.debug("Starting create_itinerary")
            request = state.get('travel_request')
            if not request:
                raise ValueError("No travel request available for itinerary creation")
                
            # Get POIs from state or use empty list
            selected_pois = state.get('explorer_output', [])
            
            # Create a travel request with the necessary parameters
            from travel_agent.models import TravelPlanRequest
            
            # Check if we already have a valid travel request
            if not isinstance(request, TravelPlanRequest):
                # Create a minimal travel request if needed
                request = TravelPlanRequest(
                    destination=request.destination,
                    duration_days=request.duration_days,
                    group_size=getattr(request, 'group_size', 1),  # Default to 1 traveler
                    budget=getattr(request, 'budget', 'mid-range'),  # Default to mid-range budget
                    interests=getattr(request, 'interests', []),
                    travel_style=getattr(request, 'travel_style', []),
                    constraints=getattr(request, 'constraints', [])
                )
            
            # Ensure start_date and end_date are set
            from datetime import date, timedelta
            if not request.start_date:
                request.start_date = date.today()
            if not request.end_date and request.duration_days:
                request.end_date = request.start_date + timedelta(days=request.duration_days - 1)
            
            # Debug: Print the selected_pois before passing to itinerary agent
            print(f"DEBUG: Selected POIs before passing to itinerary agent: {selected_pois}")
            print(f"DEBUG: Number of POIs: {len(selected_pois) if selected_pois else 0}")
            
            # Convert POIs to dictionaries if they're objects
            pois_to_pass = []
            if selected_pois:
                for poi in selected_pois:
                    if hasattr(poi, 'dict'):
                        pois_to_pass.append(poi.dict())
                    elif isinstance(poi, dict):
                        pois_to_pass.append(poi)
                    else:
                        print(f"WARNING: Skipping invalid POI: {poi}")
            
            print(f"DEBUG: Number of POIs after conversion: {len(pois_to_pass)}")
            
            # Use the itinerary agent to create an itinerary
            itinerary = await itinerary_agent.process(
                travel_request=request,
                selected_pois=pois_to_pass or []
            )
            
            # Create a new state with only the updated fields
            return {
                **state,
                'current_step': 'create_itinerary',
                'itinerary_output': itinerary,
                'messages': [
                    *state.get('messages', []),
                    AIMessage(
                        content=f"Created a {request.duration_days}-day itinerary for your trip to {request.destination} with {len(pois_to_pass) if pois_to_pass else 'no'} points of interest"
                    )
                ]
            }
            
        except Exception as e:
            error_msg = f"Error creating itinerary: {str(e)}"
            logger.error(error_msg)
            return {
                **state,
                'current_step': 'create_itinerary',
                'error': error_msg
            }
    
    async def format_output(state: AgentState) -> AgentState:
        """Format the final output for display."""
        try:
            logger.debug("Starting format_output")
            itinerary = state.get('itinerary_output')
            if not itinerary:
                logger.error("No itinerary available to format")
                return {
                    **state,
                    'current_step': 'format_output',
                    'formatter_output': "## 🎯 No itinerary data available. Let's start planning your adventure!",
                    'error': "No itinerary available to format"
                }
                
            # Get format type from state or use default
            format_type = state.get('output_format', 'markdown')
            
            # Debug logging
            logger.debug(f"Formatting itinerary with format: {format_type}")
            logger.debug(f"Itinerary type: {type(itinerary)}")
            
            # Ensure we have a valid itinerary with daily_plans
            if hasattr(itinerary, 'daily_plans') and not itinerary.daily_plans:
                logger.error("Itinerary has no daily plans")
                return {
                    **state,
                    'current_step': 'format_output',
                    'formatter_output': "## 🎯 No daily plans found in the itinerary. Let's try again!",
                    'error': "No daily plans in itinerary"
                }
            
            # Use the formatter agent to format the output
            try:
                formatted = await formatter_agent.format_itinerary(
                    itinerary=itinerary,
                    output_format=format_type
                )
                logger.debug(f"Formatted output type: {type(formatted)}")
                logger.debug(f"Formatted output preview: {str(formatted)[:200]}..." if formatted else "No formatted output")
                
                # Create a new state with the formatted output
                new_state = {
                    **state,
                    'current_step': 'format_output',
                    'formatter_output': formatted if formatted else "## 🎯 No formatted output available",
                    'messages': [
                        *state.get('messages', []),
                        AIMessage(
                            content=f"Formatted your itinerary as {format_type.upper()}"
                        )
                    ]
                }
                
                logger.debug(f"New state keys: {new_state.keys()}")
                logger.debug(f"Formatted output in state: {'formatter_output' in new_state}")
                
                return new_state
                
            except Exception as format_error:
                logger.error(f"Error in formatter agent: {str(format_error)}")
                # Fallback to basic formatting
                try:
                    if hasattr(itinerary, 'daily_plans') and itinerary.daily_plans:
                        formatted = "# Your Travel Itinerary\n\n"
                        for day in itinerary.daily_plans:
                            formatted += f"## Day {getattr(day, 'day', '?')}\n"
                            for activity in getattr(day, 'activities', []):
                                name = getattr(activity, 'name', 'Unnamed Activity')
                                start = getattr(activity, 'start_time', '')
                                end = getattr(activity, 'end_time', '')
                                desc = getattr(activity, 'description', '')
                                
                                if hasattr(start, 'strftime'):
                                    start = start.strftime('%I:%M %p')
                                if hasattr(end, 'strftime'):
                                    end = end.strftime('%I:%M %p')
                                    
                                formatted += f"- **{start} - {end}**: {name}\n"
                                if desc:
                                    formatted += f"  {desc}\n"
                                formatted += "\n"
                        
                        return {
                            **state,
                            'current_step': 'format_output',
                            'formatter_output': formatted,
                            'messages': [
                                *state.get('messages', []),
                                AIMessage(content="Used fallback formatter for your itinerary")
                            ]
                        }
                except Exception as fallback_error:
                    logger.error(f"Fallback formatting failed: {str(fallback_error)}")
                    
                # If all else fails, return a helpful error message
                return {
                    **state,
                    'current_step': 'format_output',
                    'formatter_output': "## 🎯 We had trouble formatting your itinerary. Here's what we know:\n\n" + \
                                     f"- Destination: {getattr(itinerary, 'destination', 'Unknown')}\n" + \
                                     f"- Duration: {len(getattr(itinerary, 'daily_plans', []))} days\n\n" + \
                                     "Please try again or provide more specific details about your trip.",
                    'error': f"Formatting failed: {str(format_error)}"
                }
            
        except Exception as e:
            error_msg = f"Error in format_output: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                **state,
                'current_step': 'format_output',
                'formatter_output': f"## ❌ Error\n\nWe encountered an error while formatting your itinerary.\n\nError: {str(e)}",
                'error': error_msg
            }
    
    # Create the workflow
    workflow = StateGraph(AgentState)
    
    # Add nodes to workflow
    workflow.add_node("parse_user_input", parse_user_input)
    workflow.add_node("explore_pois", explore_pois)
    workflow.add_node("plan_budget", plan_budget)
    workflow.add_node("create_itinerary", create_itinerary)
    workflow.add_node("format_output", format_output)
    
    # Set the entry point
    workflow.set_entry_point("parse_user_input")
    
    # After parsing user input, explore POIs
    workflow.add_edge("parse_user_input", "explore_pois")
    
    # After exploring POIs, plan the budget
    workflow.add_edge("explore_pois", "plan_budget")
    
    # After planning the budget, create the itinerary
    workflow.add_edge("plan_budget", "create_itinerary")
    
    # After creating the itinerary, format the output
    workflow.add_edge("create_itinerary", "format_output")
    
    # The format_output node is the end of the workflow
    workflow.set_finish_point("format_output")
    
    return workflow

# Create a default workflow instance
travel_planner_workflow = create_travel_planner_workflow().compile()

def run_workflow(user_input: str):
    """Run the travel planning workflow with the provided user input.
    
    Args:
        user_input: Natural language description of the trip
    """
    print("Running travel planning workflow...")
    
    print(f"Processing request: {user_input.strip()}")
    
    # Run the workflow
    result = asyncio.run(enhanced_run_workflow(
        user_input=user_input,
        user_id="test_user_123",
        output_format="markdown",
        save_to_file=True
    ))
    
    print("\nWorkflow completed!")
    print(f"Status: {result.get('status', 'unknown')}")
    
    if result.get('status') == 'success':
        print("\nGenerated Itinerary:")
        print("-" * 40)
        print(result.get('formatted_output', 'No output generated'))
    
    return result

# Example usage with enhanced features
async def enhanced_run_workflow(
    user_input: str,
    user_id: str = "default",
    output_format: str = "markdown",
    save_to_file: bool = False
) -> Dict[str, Any]:
    """Run the enhanced travel planning workflow.
    
    Args:
        user_input: Natural language description of the trip
        user_id: Unique user ID for personalization
        output_format: Output format (markdown, json, text, html)
        save_to_file: Whether to save the output to a file
        
    Returns:
        Dictionary with the workflow results
    """
    # Initialize the workflow
    workflow = create_travel_planner_workflow()
    
    # Create initial state
    state = AgentState(
        user_id=user_id,
        user_input=user_input,
        output_format=output_format
    )
    
    try:
        # Execute the workflow
        result = await workflow.ainvoke({"values": state})
        final_state = result["values"]
        
        # Save to file if requested
        if save_to_file and hasattr(final_state, 'formatted_itinerary'):
            filename = f"itinerary_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_format}"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(final_state.formatted_itinerary)
            print(f"Itinerary saved to {filename}")
            
            # Save calendar view if available
            if hasattr(final_state, 'calendar_view') and final_state.calendar_view:
                cal_filename = f"itinerary_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                with open(cal_filename, 'w', encoding='utf-8') as f:
                    f.write(final_state.calendar_view)
                print(f"Calendar view saved to {cal_filename}")
        
        return {
            "status": "success",
            "itinerary": final_state.formatted_itinerary,
            "calendar_view": getattr(final_state, 'calendar_view', None),
            "messages": final_state.messages,
            "user_cart": final_state.user_cart.get_summary() if hasattr(final_state, 'user_cart') else None
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "messages": getattr(state, 'messages', []) + [f"Workflow failed: {str(e)}"]
        }
if __name__ == "__main__":
    async def run_workflow():
        # Initialize the state
        initial_state = {
            "user_input": "I want to visit Paris for 3 days in July. I'm interested in art and history.",
            "output_format": "markdown",  # Can be markdown, json, text, or html
            "travel_request": None,
            "suggested_pois": [],
            "selected_pois": [],
            "itinerary": [],
            "formatted_itinerary": "",
            "messages": [],
            "status": "planning",
            "error": None
        }
        
        # Run the workflow
        print("Starting travel planning workflow...")
        result = await travel_planner_workflow.ainvoke(initial_state)
        
        # Print the result
        print("\nWorkflow completed!")
        print(f"Status: {result['status']}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
        
        if result.get('itinerary'):
            print("\nGenerated Itinerary:")
            for day in result['itinerary']:
                print(f"\nDay {day['day_number']} ({day['date']}):")
                for activity in day['activities']:
                    print(f"  {activity['start_time'].strftime('%H:%M')}-{activity['end_time'].strftime('%H:%M')} {activity['name']} ({activity['location']})")
        
        print("\nMessages:")
        for msg in result.get('messages', []):
            print(f"- {msg}")
    
    import asyncio
    asyncio.run(run_workflow())
