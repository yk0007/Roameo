"""Planner Agent for parsing user input into structured travel plan requests."""
from typing import Optional
from datetime import date, timedelta
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from .base import BaseAgent
from .models import TravelPlanRequest

class PlannerAgent(BaseAgent):
    """Agent responsible for parsing user input into structured travel plan requests."""
    
    def __init__(self, model_name: str = "openai/gpt-oss-20b", temperature: float = 0.3):
        """Initialize the PlannerAgent with the specified Groq model.
        
        Args:
            model_name: The name of the Groq model to use.
            temperature: The temperature for the LLM.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = self._initialize_llm()
        self.parser = PydanticOutputParser(pydantic_object=TravelPlanRequest)
        
        # Define the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert travel planner. Your task is to extract key information from the user's 
            travel request and structure it into a standardized format.
            
            {format_instructions}
            
            If any information is missing, make reasonable assumptions and note them in the constraints.
            """),
            ("human", "{input}")
        ]).partial(
            format_instructions=self.parser.get_format_instructions()
        )
        
        # Create the chain
        self.chain = self.prompt | self.llm | self.parser
    
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
                temperature=self.temperature,
                model_kwargs={"top_p": 0.8}
            )
        else:
            # Default to Groq for other models
            return ChatGroq(
                model_name=self.model_name,
                temperature=self.temperature,
                model_kwargs={"top_p": 0.8}
            )
    
    async def process(self, user_input: str) -> TravelPlanRequest:
        """Process the user input and return a structured travel plan request.
        
        Args:
            user_input: The raw user input describing their travel plans.
            
        Returns:
            A structured TravelPlanRequest object with all required fields.
        """
        try:
            print(f"\n=== DEBUG: Processing user input: {user_input}")
            
            # Use the chain to process the input
            print("=== DEBUG: Invoking LLM chain...")
            result = await self.chain.ainvoke({"input": user_input})
            print(f"=== DEBUG: Raw result from LLM: {result}")
            
            # Debug: Print all attributes of the result
            print(f"=== DEBUG: Result attributes: {dir(result)}")
            
            # Ensure all required fields have values
            if not hasattr(result, 'travel_style') or not result.travel_style:
                print("=== DEBUG: Setting default travel_style")
                result.travel_style = ["cultural"]
            if not hasattr(result, 'budget') or not result.budget:
                print("=== DEBUG: Setting default budget")
                result.budget = "mid-range"
            if not hasattr(result, 'interests') or not result.interests:
                print("=== DEBUG: Setting default interests")
                result.interests = ["sightseeing"]
                
            print(f"=== DEBUG: Final result before return: {result}")
            print(f"=== DEBUG: Final interests: {getattr(result, 'interests', 'NOT FOUND')}")
            
            return result
            
        except Exception as e:
            print(f"Error in process: {str(e)}")
            # Fallback to a more robust parsing approach if needed
            return self._fallback_parse(user_input)
    
    def _fallback_parse(self, user_input: str) -> TravelPlanRequest:
        """Fallback parsing logic if the main parsing fails."""
        from .models import TravelStyle, BudgetLevel  # Import here to avoid circular imports
        
        print("\n=== DEBUG: Entering _fallback_parse")
        
        # Extract basic information from user input if possible
        destination = ""
        duration_days = 7  # Default duration
        
        # Try to extract destination
        if " to " in user_input:
            # Simple heuristic to extract destination
            parts = user_input.split(" to ", 1)
            if len(parts) > 1:
                destination = parts[1].split(" ", 1)[0].strip(".,!?")
        
        # Try to extract duration
        for word in user_input.split():
            if word.isdigit() and 1 <= int(word) <= 30:  # Reasonable range for trip duration
                duration_days = int(word)
                break
        
        print(f"=== DEBUG: Creating TravelPlanRequest with:")
        print(f"  destination: {destination or 'Unknown Destination'}")
        print(f"  duration_days: {duration_days}")
        print("  travel_style: ['cultural']")
        print("  budget: BudgetLevel.MID_RANGE")
        print("  interests: ['sightseeing']")
        
        # Set default values for all required fields with proper typing
        # Using string values that match the TravelStyle enum
        request = TravelPlanRequest(
            destination=destination or "Unknown Destination",
            duration_days=duration_days,
            travel_style=["cultural"],  # Using a valid string value from TravelStyle
            budget=BudgetLevel.MID_RANGE,
            interests=["sightseeing"],
            start_date=date.today() + timedelta(days=30),  # Default to 30 days from now
            end_date=date.today() + timedelta(days=30 + duration_days - 1),
            origin="Unknown Origin",
            constraints=["Incomplete information provided"]
        )
        
        print(f"=== DEBUG: Created TravelPlanRequest: {request}")
        print(f"=== DEBUG: Request fields: {request.__fields__}")
        print(f"=== DEBUG: Request interests: {getattr(request, 'interests', 'NOT FOUND')}")
        
        return request

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_planner_agent():
        agent = PlannerAgent()
        test_input = """
        I want to go to Paris for 5 days with my wife. We love art and good food. 
        Our budget is around $3000 and we prefer mid-range accommodations.
        """
        
        result = await agent.process(test_input)
        print("Parsed Travel Plan Request:")
        print(f"Destination: {result.destination}")
        print(f"Duration: {result.duration_days} days")
        print(f"Travel Style: {', '.join(result.travel_style) if result.travel_style else 'Not specified'}")
        print(f"Budget: {result.budget or 'Not specified'}")
        print(f"Constraints: {', '.join(result.constraints) if result.constraints else 'None'}")
    
    asyncio.run(test_planner_agent())
