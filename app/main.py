import asyncio
import os
import sys
from google.adk.runners import InMemoryRunner
from google.adk.events.request_input import RequestInput
from google.genai import types

# Add the parent directory of 'app' to sys.path so we can run this directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent import app

async def run_cli():
    print("==================================================")
    print("🍽️  Welcome to Meal Concierge Multi-Agent CLI 🍽️")
    print("==================================================")
    
    # Initialize the runner
    runner = InMemoryRunner(app=app)
    user_id = "chef_user"
    session = await runner.session_service.create_session(app_name="app", user_id=user_id)
    session_id = session.id
    
    print("\nHow can I help you today?")
    print("👉 Ask to 'plan my week' or write a rating like 'I loved the Spaghetti Bolognese'.")
    query = input("\nUser: ").strip()
    if not query:
        print("Goodbye!")
        return

    # Set up initial input message
    new_message = types.Content(role="user", parts=[types.Part.from_text(text=query)])
    
    while True:
        interrupted = False
        interrupt_id = None
        interrupt_msg = "Please respond to approval request:"
        
        # Run the workflow
        async for event in runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=new_message
        ):
            # 1. Print visual content (e.g. proposed menus or shopping lists)
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        print(part.text, end="", flush=True)
                    
                    # 2. Check if the node paused for user feedback (RequestInput)
                    # ADK wraps RequestInput as an adk_request_input FunctionCall event
                    if part.function_call and part.function_call.name == "adk_request_input":
                        interrupted = True
                        interrupt_id = part.function_call.id
                        args = part.function_call.args or {}
                        if "message" in args:
                            interrupt_msg = args["message"]
        
        # If the workflow paused for a review/approval, prompt the user and resume
        if interrupted:
            user_reply = input(f"\nPrompt: {interrupt_msg}\nReply: ").strip()
            
            # Resume using a FunctionResponse part as expected by ADK
            new_message = types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="adk_request_input",
                            id=interrupt_id,
                            response={"result": user_reply}
                        )
                    )
                ]
            )
            print("\n⏳ Processing adjustments...\n")
        else:
            # Workflow completed successfully
            break

    print("\n==================================================")
    print("✨ Meal Concierge Multi-Agent Workflow Completed ✨")
    print("==================================================")

if __name__ == "__main__":
    try:
        asyncio.run(run_cli())
    except RuntimeError:
        # Fallback for environments with active loop
        loop = asyncio.get_event_loop()
        loop.run_until_complete(run_cli())
