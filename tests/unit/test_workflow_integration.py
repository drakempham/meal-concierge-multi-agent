import pytest
import os
from google.adk.runners import InMemoryRunner
from google.adk.events.request_input import RequestInput
from google.genai import types

from app.agent import app
from app.memory import load_profile

@pytest.mark.asyncio
async def test_workflow_plan_and_approve():
    """Tests the full meal planning workflow, including the user review interruption and approval resumption."""
    # Ensure GEMINI_API_KEY is available for this test
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("Skipping integration test: GEMINI_API_KEY environment variable not set.")
        
    runner = InMemoryRunner(app=app)
    user_id = "test_user_integration"
    
    # 1. Create a session
    session = await runner.session_service.create_session(app_name="app", user_id=user_id)
    session_id = session.id
    
    # 2. Initial user query
    new_message = types.Content(role="user", parts=[types.Part.from_text(text="plan my week")])
    
    interrupted = False
    interrupt_id = None
    
    # Run the workflow - should pause at user_review_node
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call and part.function_call.name == "adk_request_input":
                    interrupted = True
                    interrupt_id = part.function_call.id
                    
    assert interrupted, "Workflow should have paused for user review"
    assert interrupt_id == "user_approval_0", "Interruption ID should be 'user_approval_0'"
    
    # 3. Resume the workflow by approving ('yes')
    resume_message = types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    name="adk_request_input",
                    id=interrupt_id,
                    response={"result": "yes"}
                )
            )
        ]
    )
    
    completed = False
    grocery_list_found = False
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=resume_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and "grocery list" in part.text.lower():
                    grocery_list_found = True
        
        # When workflow completes, runner yields events. The final output is returned
        if event.output is not None:
            completed = True
            
    assert completed, "Workflow should have completed after approval"
    assert grocery_list_found, "Grocery list should have been generated and displayed"

@pytest.mark.asyncio
async def test_workflow_feedback_rotation():
    """Tests the feedback route that parses sentiment and updates the favorites rotation in user_profile.json."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("Skipping integration test: GEMINI_API_KEY environment variable not set.")
        
    runner = InMemoryRunner(app=app)
    user_id = "test_user_integration"
    
    session = await runner.session_service.create_session(app_name="app", user_id=user_id)
    session_id = session.id
    
    # Send feedback about a meal
    feedback_text = "I really loved the Spaghetti Bolognese we had last night, it was amazing!"
    new_message = types.Content(role="user", parts=[types.Part.from_text(text=feedback_text)])
    
    completed = False
    feedback_processed = False
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and "memory update" in part.text.lower():
                    feedback_processed = True
        if event.output is not None:
            completed = True
            
    assert completed, "Feedback workflow should have completed"
    assert feedback_processed, "Memory update message should be outputted"
    
    # Read profile to verify that Spaghetti Bolognese is added to the rotation list
    profile = load_profile()
    rotation_names = [r["name"].strip().lower() for r in profile.get("recipe_rotation", [])]
    assert "spaghetti bolognese" in rotation_names, "Spaghetti Bolognese should be in the recipe rotation favorites list"

@pytest.mark.asyncio
async def test_workflow_feedback_rotation_negative():
    """Tests the feedback route for negative sentiment to verify it removes/logs removal of a recipe from rotation."""
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("Skipping integration test: GEMINI_API_KEY environment variable not set.")
        
    runner = InMemoryRunner(app=app)
    user_id = "test_user_integration"
    
    session = await runner.session_service.create_session(app_name="app", user_id=user_id)
    session_id = session.id
    
    # 1. Ensure Spaghetti Bolognese is in the list first
    from app.memory import add_recipe_to_rotation
    add_recipe_to_rotation({
        "name": "Spaghetti Bolognese",
        "ingredients": {"pasta_g": 500, "ground_beef_g": 500},
        "prep_time_minutes": 10,
        "cook_time_minutes": 30,
        "cooking_style": "Stovetop",
        "protein_g": 125,
        "fat_g": 60,
        "calories": 2000
    })
    
    # Send negative feedback about the meal
    feedback_text = "I hated the Spaghetti Bolognese we cooked, it was way too greasy and we didn't like it."
    new_message = types.Content(role="user", parts=[types.Part.from_text(text=feedback_text)])
    
    completed = False
    feedback_processed = False
    
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=new_message
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text and "removed" in part.text.lower():
                    feedback_processed = True
        if event.output is not None:
            completed = True
            
    assert completed, "Feedback workflow should have completed"
    assert feedback_processed, "Memory removal/update confirmation should be outputted"
    
    # Read profile to verify that Spaghetti Bolognese is removed from the rotation list
    profile = load_profile()
    rotation_names = [r["name"].strip().lower() for r in profile.get("recipe_rotation", [])]
    assert "spaghetti bolognese" not in rotation_names, "Spaghetti Bolognese should have been removed from the recipe rotation favorites list"
