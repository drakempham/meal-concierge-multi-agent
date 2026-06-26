import datetime
from typing import Dict, Any, List
from app.memory import load_profile

# Mock Weather Forecast database for the upcoming week
MOCK_WEATHER = {
    "monday": "Sunny, 78°F. Clear skies, calm wind. (Great outdoor cooking weather)",
    "tuesday": "Rainy, 65°F. Heavy rain, high humidity. (Indoor cooking recommended, grilling disabled)",
    "wednesday": "Sunny, 80°F. Clear skies. (Great outdoor cooking weather)",
    "thursday": "Thunderstorms, 60°F. Windy and wet. (Indoor cooking recommended)",
    "friday": "Cloudy, 72°F. Moderate clouds, light breeze. (Indoor cooking recommended)",
    "saturday": "Sunny, 82°F. Perfect summer day. (Perfect for grilling outdoors)",
    "sunday": "Rainy, 64°F. Light rain in the afternoon. (Indoor cooking recommended)"
}

# Mock Calendar Database for the upcoming week
# Headcount: Mon-Thu kids are home (4 people). Fri-Sun kids are at joint custody (2 people).
MOCK_CALENDAR = {
    "monday": {
        "headcount": 4,
        "events": ["Kid 1 Soccer practice 5:00 PM - 6:30 PM", "Parents normal work hours"],
        "recommended_dinner_time": "7:30 PM (Late dinner)",
        "prep_time_constraint": "Under 30 minutes (Busy evening)"
    },
    "tuesday": {
        "headcount": 4,
        "events": ["Standard day. No sports or late meetings."],
        "recommended_dinner_time": "6:30 PM (Standard time)",
        "prep_time_constraint": "No limit"
    },
    "wednesday": {
        "headcount": 4,
        "events": ["Husband has late office meeting until 7:00 PM"],
        "recommended_dinner_time": "7:30 PM (Late dinner)",
        "prep_time_constraint": "Under 30 minutes"
    },
    "thursday": {
        "headcount": 4,
        "events": ["Kid 2 Swimming class 5:00 PM - 6:00 PM"],
        "recommended_dinner_time": "7:00 PM (Late dinner)",
        "prep_time_constraint": "Under 30 minutes"
    },
    "friday": {
        "headcount": 2,
        "events": ["Kids away for joint custody weekend. Only Husband & Wife at home."],
        "recommended_dinner_time": "6:30 PM (Standard time)",
        "prep_time_constraint": "No limit"
    },
    "saturday": {
        "headcount": 2,
        "events": ["Kids away. Only Husband & Wife at home."],
        "recommended_dinner_time": "6:30 PM (Standard time)",
        "prep_time_constraint": "No limit"
    },
    "sunday": {
        "headcount": 2,
        "events": ["Kids away. Only Husband & Wife at home."],
        "recommended_dinner_time": "6:30 PM (Standard time)",
        "prep_time_constraint": "No limit"
    }
}

def get_weather_forecast(day_name: str) -> str:
    """Returns the weather forecast for a given day of the week (e.g. 'Monday').
    
    Args:
        day_name: Name of the day of the week (Monday - Sunday).
    """
    day_key = day_name.strip().lower()
    if day_key not in MOCK_WEATHER:
        # Default fallback to a generic forecast
        return "Partly cloudy, 70°F. (Indoor or outdoor cooking possible)"
    return MOCK_WEATHER[day_key]

def get_family_calendar(day_name: str) -> Dict[str, Any]:
    """Returns the family's schedule events, headcount, and dinner time constraints for a day.
    
    Args:
        day_name: Name of the day of the week (Monday - Sunday).
    """
    day_key = day_name.strip().lower()
    if day_key not in MOCK_CALENDAR:
        return {
            "headcount": 4,
            "events": ["Standard day"],
            "recommended_dinner_time": "6:30 PM",
            "prep_time_constraint": "No limit"
        }
    return MOCK_CALENDAR[day_key]

def get_pantry_inventory() -> Dict[str, int]:
    """Returns the current quantities of ingredients available in the pantry/fridge."""
    try:
        profile = load_profile()
        return profile.get("pantry", {})
    except Exception:
        # Return fallback mock database if file reading fails
        return {
            "chicken_breast_g": 300,
            "pasta_g": 500,
            "cheese_g": 200,
            "tomato_sauce_g": 600
        }
