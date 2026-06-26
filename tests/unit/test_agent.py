import pytest
from app.tools import get_weather_forecast, get_family_calendar, get_pantry_inventory
from app.memory import load_profile
from app.agent import grocery_list_builder, WeeklyMenu, DailyMeal, IngredientItem

def test_weather_tool():
    """Verifies that the weather tool returns expected forecasts for specific days."""
    monday_forecast = get_weather_forecast("Monday")
    assert "Sunny" in monday_forecast
    
    tuesday_forecast = get_weather_forecast("Tuesday")
    assert "Rainy" in tuesday_forecast
    
    invalid_forecast = get_weather_forecast("InvalidDay")
    assert "cloudy" in invalid_forecast.lower()

def test_calendar_tool():
    """Verifies that the calendar tool returns correct schedules and headcounts."""
    monday_cal = get_family_calendar("Monday")
    assert monday_cal["headcount"] == 4
    assert "Soccer practice" in monday_cal["events"][0]
    assert "7:30 PM" in monday_cal["recommended_dinner_time"]
    
    friday_cal = get_family_calendar("Friday")
    assert friday_cal["headcount"] == 2
    assert "Kids away" in friday_cal["events"][0]

def test_pantry_loading():
    """Verifies that the pantry inventory tool loads data matching the profile configuration."""
    profile = load_profile()
    pantry = get_pantry_inventory()
    assert isinstance(pantry, dict)
    assert pantry["flour_g"] == profile["pantry"]["flour_g"]

def test_grocery_list_builder():
    """Verifies that the grocery list builder correctly subtracts existing pantry items."""
    # Create a mock WeeklyMenu
    mock_menu = WeeklyMenu(
        meals=[
            DailyMeal(
                day="Monday",
                meal_name="Grilled Chicken Souvlaki",
                prep_time_minutes=15,
                cooking_style="Grill",
                estimated_protein_g=140,
                estimated_fat_g=35,
                estimated_calories=900,
                recipe_source="rotation",
                ingredients=[
                    IngredientItem(name="chicken_breast_g", amount=600), # Pantry has 300g -> needs 300g
                    IngredientItem(name="lemon", amount=1),              # Pantry has 0 -> needs 1
                    IngredientItem(name="butter_g", amount=50)            # Pantry has 250g -> needs 0
                ]
            )
        ],
        rationale="Test plan"
    )
    
    # Access the underlying callable function of the FunctionNode in ADK 2.0
    # In ADK 2.0, decorated functions become FunctionNode instances, and their original function is in _func
    builder_func = grocery_list_builder._func if hasattr(grocery_list_builder, "_func") else grocery_list_builder
    
    # Run the builder
    # Since the node is a generator function, we convert its yields to a list and find the output event.
    events = list(builder_func(mock_menu))
    output_event = next(e for e in events if getattr(e, "output", None) is not None)
    output = output_event.output
    
    assert "shopping_list" in output
    shopping_list = output["shopping_list"]
    
    # Check expected shopping requirements
    assert shopping_list["chicken_breast_g"] == 300  # 600 required - 300 in pantry
    assert shopping_list["lemon"] == 1               # 1 required - 0 in pantry
    assert "butter_g" not in shopping_list           # 50 required - 250 in pantry -> plenty available
