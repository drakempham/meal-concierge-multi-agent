import os
import json
from typing import Dict, Any, List

PROFILE_PATH = os.path.join(os.path.dirname(__file__), "user_profile.json")

def load_profile() -> Dict[str, Any]:
    """Loads the user profile from user_profile.json."""
    if not os.path.exists(PROFILE_PATH):
        raise FileNotFoundError(f"User profile configuration not found at {PROFILE_PATH}")
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_profile(profile: Dict[str, Any]) -> None:
    """Saves the user profile to user_profile.json."""
    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

def add_recipe_to_rotation(recipe: Dict[str, Any]) -> bool:
    """Appends a new recipe to the rotation list if it does not already exist."""
    profile = load_profile()
    rotation = profile.setdefault("recipe_rotation", [])
    
    # Check if already exists by name
    for r in rotation:
        if r["name"].strip().lower() == recipe["name"].strip().lower():
            return False
            
    rotation.append(recipe)
    save_profile(profile)
    return True

def update_pantry(pantry_updates: Dict[str, int]) -> None:
    """Updates specific pantry ingredients with new quantities."""
    profile = load_profile()
    pantry = profile.setdefault("pantry", {})
    pantry.update(pantry_updates)
    save_profile(profile)

def remove_recipe_from_rotation(recipe_name: str) -> bool:
    """Removes a recipe from the rotation list if it exists."""
    profile = load_profile()
    rotation = profile.setdefault("recipe_rotation", [])
    
    initial_len = len(rotation)
    profile["recipe_rotation"] = [
        r for r in rotation if r["name"].strip().lower() != recipe_name.strip().lower()
    ]
    
    if len(profile["recipe_rotation"]) < initial_len:
        save_profile(profile)
        return True
    return False

def add_recipe_to_dislikes(recipe_name: str) -> bool:
    """Appends a recipe name to the disliked_recipes list if it does not already exist."""
    profile = load_profile()
    dislikes = profile.setdefault("disliked_recipes", [])
    name_clean = recipe_name.strip().lower()
    
    # Check if already exists
    if name_clean in [d.strip().lower() for d in dislikes]:
        return False
        
    dislikes.append(recipe_name.strip())
    save_profile(profile)
    return True
