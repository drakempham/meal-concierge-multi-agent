# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import base64
from typing import Dict, Any, List, Optional, Literal
from pydantic import BaseModel, Field

from google.adk.workflow import Workflow, node, START
from google.adk.events.event import Event
from google.adk.events.request_input import RequestInput
from google.adk.agents.context import Context
from google.adk.agents import LlmAgent
from google.adk.models import Gemini
from google.adk.apps import App, ResumabilityConfig
from google.genai import types

from app.tools import get_weather_forecast, get_family_calendar, get_pantry_inventory
from app.memory import load_profile, add_recipe_to_rotation, remove_recipe_from_rotation

# Ensure target environment variables are set for local execution
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "mock-project")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "False")

import logging
import contextlib
import asyncio

logger = logging.getLogger(__name__)

class FallbackGemini(Gemini):
    fallback_models: List[str] = [
        "gemini-3.5-flash",          # RPD: 20 — ưu tiên 1 (chất lượng cao nhất)
        "gemini-3.1-flash-lite",     # RPD: 500 — ưu tiên 2 (RPD rất lớn, stable)
        "gemini-2.5-flash-lite",     # RPD: 20 — ưu tiên 3
        "gemini-3-flash-preview",    # RPD: 20 — ưu tiên 4 (preview)
        "gemini-2.5-flash",          # RPD: 20 — ưu tiên 5
        "gemini-2.0-flash-lite",     # RPD: cao — backup
        "gemini-2.0-flash",          # RPD: cao — backup cuối cùng
    ]
    
    def _should_fallback(self, e: Exception) -> bool:
        from google.adk.models.google_llm import _ResourceExhaustedError
        if isinstance(e, _ResourceExhaustedError):
            return True
        err_str = str(e).upper()
        # Fallback on quota (429), server unavailable (503), model not found (404), or bad request (400)
        return any(msg in err_str for msg in [
            "429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "HIGH DEMAND", "LIMIT", "QUOTA",
            "404", "NOT_FOUND", "400", "INVALID_ARGUMENT"
        ])
    
    # Timeout (seconds) per model attempt before switching to next fallback
    request_timeout: int = 45

    async def generate_content_async(self, llm_request, stream: bool = False):
        # Save the original model request name
        original_model = llm_request.model or self.model
        
        # Try models in sequence
        models_to_try = [original_model] + [m for m in self.fallback_models if m != original_model]
        
        last_error = None
        for model in models_to_try:
            llm_request.model = model
            logger.info(f"FallbackGemini: Trying model '{model}' for generate_content (timeout={self.request_timeout}s)...")
            try:
                # Collect all responses from the async generator within the timeout window
                async def _collect():
                    results = []
                    async for response in super(FallbackGemini, self).generate_content_async(llm_request, stream=stream):
                        results.append(response)
                    return results

                responses = await asyncio.wait_for(_collect(), timeout=self.request_timeout)
                for response in responses:
                    yield response
                return  # Success!
            except asyncio.TimeoutError:
                logger.warning(f"FallbackGemini: Model '{model}' timed out after {self.request_timeout}s. Trying next fallback model...")
                last_error = asyncio.TimeoutError(f"Model '{model}' timed out after {self.request_timeout}s")
                continue
            except Exception as e:
                if self._should_fallback(e):
                    logger.warning(f"FallbackGemini: Model '{model}' failed with retryable/quota error ({e}). Trying next fallback model...")
                    last_error = e
                    continue
                else:
                    # For other exceptions (like authentication errors), raise immediately
                    raise e
                    
        # If all models failed, raise the last error
        if last_error:
            raise last_error

    @contextlib.asynccontextmanager
    async def connect(self, llm_request):
        original_model = llm_request.model or self.model
        models_to_try = [original_model] + [m for m in self.fallback_models if m != original_model]
        
        last_error = None
        for model in models_to_try:
            llm_request.model = model
            logger.info(f"FallbackGemini: Trying model '{model}' for connect...")
            try:
                async with super().connect(llm_request) as conn:
                    yield conn
                return  # Success!
            except Exception as e:
                if self._should_fallback(e):
                    logger.warning(f"FallbackGemini: Model '{model}' failed on connect ({e}). Trying next fallback...")
                    last_error = e
                    continue
                else:
                    raise e
        if last_error:
            raise last_error

# Define Gemini model using local credentials / GEMINI_API_KEY
# IMPORTANT: attempts=1 disables SDK-level retries so that 503/429 errors bubble up
# immediately to our FallbackGemini logic instead of waiting for multiple SDK retries.
# FallbackGemini handles all retry/fallback logic itself with a 20-second per-model timeout.
model_instance = FallbackGemini(
    model="gemini-3-flash-preview",
    retry_options=types.HttpRetryOptions(
        attempts=1,              # No SDK-level retries — FallbackGemini handles it
        initialDelay=0.5,        # Short initial delay if any retry happens
        maxDelay=2.0,            # Max delay cap between retries
    ),
    request_timeout=20,          # Switch to next model after 20 seconds
)

# ==========================================
# 1. Pydantic Schemas for Structured I/O
# ==========================================

class UserRequest(BaseModel):
    query: str = Field(description="The input query from the user")

class IngredientItem(BaseModel):
    name: str = Field(description="Name of the ingredient, e.g. chicken_breast_g, lemon")
    amount: int = Field(description="Quantity required in grams, ml, or units")

class DailyMeal(BaseModel):
    day: str = Field(description="Name of the day, e.g., Monday")
    meal_name: str = Field(description="Name of the meal suggested")
    prep_time_minutes: int = Field(description="Estimated preparation time in minutes")
    cooking_style: str = Field(description="Style of cooking, e.g., Grill, Slow Cooker, Stovetop, Oven")
    estimated_protein_g: int = Field(description="Estimated protein content in grams")
    estimated_fat_g: int = Field(description="Estimated fat content in grams")
    estimated_calories: int = Field(description="Estimated calorie count for the meal")
    recipe_source: str = Field(description="Source of the recipe, either 'rotation' (from favorites list) or 'new' (suggested new dish)")
    ingredients: List[IngredientItem] = Field(description="List of required ingredients and their amounts")

class WeeklyMenu(BaseModel):
    meals: List[DailyMeal] = Field(description="List of daily meal suggestions for 7 days")
    rationale: str = Field(description="Short rationale explaining how weather, schedules, and preferences dictated the menu")

class WeeklyContext(BaseModel):
    days_info: List[Dict[str, Any]] = Field(description="Parsed weather, calendar schedules, and headcount per day")
    preferences: Dict[str, Any] = Field(description="Dietary restrictions, macro targets, and likes/dislikes")
    pantry: Dict[str, int] = Field(description="Current pantry/fridge inventory")
    adjustments: Optional[str] = Field(default=None, description="User-requested adjustments to the previous menu (e.g. 'Change Wednesday to pasta')")
    previous_menu: Optional[List[Dict[str, Any]]] = Field(default=None, description="The previous 7-day menu proposal that needs to be modified based on the adjustments. If provided, ONLY change what the user asked. Keep all other days exactly as they were.")

# ==========================================
# 2. Graph Nodes & LLM Agents
# ==========================================

# ---------------------------------------------------------------------------
# Embedding-based Semantic Router
# Uses Gemini Embedding API (RPD: 1000) instead of LLM (RPD: 20) for routing.
# No LLM call needed — pure vector similarity math (~100ms vs ~2s for LLM).
# ---------------------------------------------------------------------------

# Reference example sentences per intent category.
# The router embeds these once at startup, caches the centroid, then compares
# each user query against the centroid using cosine similarity.
_INTENT_EXAMPLES: Dict[str, List[str]] = {
    "feedback": [
        "We loved the Spaghetti Bolognese last night!",
        "I hated the pizza, it was too greasy",
        "The chicken dish was amazing, kids absolutely loved it",
        "Rate: the pasta was really tasty tonight",
        "I didn't like the fish dish at all",
        "The beef stew tasted terrible",
        "My family really enjoyed the tacos, great meal!",
        "That grilled chicken was delicious, please save it",
        "We cooked the recipe and it was awful",
        "Fantastic meal, everyone loved the salad",
    ],
    "plan": [
        "Plan my weekly meals please",
        "What should we eat this week?",
        "Create a 7-day meal schedule for our family",
        "Change Wednesday to pasta",
        "I want healthier meals this week",
        "Can you suggest something quick for dinner?",
        "Make me a meal plan",
        "I need a menu for the upcoming week",
        "Adjust Thursday to something easy with short prep time",
        "Generate a meal plan based on our pantry",
    ],
}

# Module-level cache for pre-computed reference centroid embeddings.
# Populated on first call, reused for all subsequent routing calls.
_embedding_centroids: Dict[str, List[float]] = {}


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors using pure Python math."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_text_sync(api_key: str, text: str) -> List[float]:
    """Call Gemini Embedding API synchronously and return the embedding vector.
    
    Tries multiple model names in order of stability:
    1. text-embedding-004 — most stable, well-supported in v1beta
    2. gemini-embedding-exp-03-07 — experimental newer model
    3. gemini-embedding-1 — newest, may require v1 API
    """
    from google import genai as _genai
    client = _genai.Client(api_key=api_key)
    
    # Try models in order of stability
    embedding_models = [
        "text-embedding-004",
        "gemini-embedding-exp-03-07",
        "gemini-embedding-1",
    ]
    last_error = None
    for model_name in embedding_models:
        try:
            result = client.models.embed_content(
                model=model_name,
                contents=text,
            )
            # Handle both SDK response formats
            if hasattr(result, "embedding") and result.embedding:
                return list(result.embedding.values)
            elif hasattr(result, "embeddings") and result.embeddings:
                return list(result.embeddings[0].values)
            else:
                raise ValueError(f"Unexpected response format from {model_name}: {dir(result)}")
        except Exception as e:
            last_error = e
            logger.debug(f"EmbeddingRouter: model '{model_name}' failed: {e}")
            continue
    
    raise RuntimeError(f"All embedding models failed. Last error: {last_error}")


def _compute_centroid(vectors: List[List[float]]) -> List[float]:
    """Compute the element-wise average (centroid) of a list of vectors."""
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


@node
async def embedding_intent_router(ctx: Context, node_input: Any) -> Event:
    """
    Embedding-based Semantic Router (replaces keyword/LLM router).

    Uses Gemini Embedding API to compute cosine similarity between the user
    query and pre-computed centroid embeddings for each intent category.
    Routes to the category with the highest similarity score.

    Quota: gemini-embedding-1 (RPD: 1000) — completely isolated from main model pool.
    Fallback: keyword matching if Embedding API is unavailable.
    """
    global _embedding_centroids

    # 1. Extract raw user query string from ADK input
    query = ""
    if hasattr(node_input, "parts") and node_input.parts:
        query = "".join(part.text for part in node_input.parts if part.text)
    elif isinstance(node_input, str):
        query = node_input
    elif isinstance(node_input, dict):
        query = node_input.get("query", str(node_input))
    query = query.strip()

    api_key = os.environ.get("GEMINI_API_KEY", "")

    try:
        # 2. Compute reference centroids on first call (cached for all subsequent calls)
        if not _embedding_centroids:
            logger.info("EmbeddingRouter: Computing reference centroids (one-time startup cost)...")
            for intent, examples in _INTENT_EXAMPLES.items():
                vectors = await asyncio.to_thread(
                    lambda exs=examples, key=api_key: [
                        _embed_text_sync(key, ex) for ex in exs
                    ]
                )
                _embedding_centroids[intent] = _compute_centroid(vectors)
            logger.info("EmbeddingRouter: Reference centroids cached successfully.")

        # 3. Embed the user query (async, non-blocking)
        query_vec = await asyncio.to_thread(_embed_text_sync, api_key, query)

        # 4. Compute cosine similarity against each intent centroid
        scores = {
            intent: _cosine_similarity(query_vec, centroid)
            for intent, centroid in _embedding_centroids.items()
        }

        # 5. Route to the highest-scoring intent
        intent = max(scores, key=scores.get)
        logger.info(
            f"EmbeddingRouter: '{query[:60]}' → intent='{intent}' "
            f"(plan={scores['plan']:.3f}, feedback={scores['feedback']:.3f})"
        )

    except Exception as e:
        # Fallback: simple keyword matching if Embedding API is unavailable
        logger.warning(f"EmbeddingRouter: Embedding API failed ({e}). Using keyword fallback.")
        query_lower = query.lower()
        feedback_keywords = [
            "feedback", "rate", "like", "love", "loved", "tasty", "great",
            "hate", "hated", "dislike", "terrible", "awful", "disgusting",
            "amazing", "fantastic", "delicious", "enjoyed", "didn't like",
        ]
        intent = "feedback" if any(k in query_lower for k in feedback_keywords) else "plan"
        logger.info(f"EmbeddingRouter: Keyword fallback → intent='{intent}'")

    return Event(output=query, route=intent, state={"user_query": query})

@node
def collect_context(ctx: Context, node_input: Any) -> WeeklyContext:
    """Preprocesses calendar, weather, pantry, and profile memory into a single structured context."""
    profile = load_profile()
    pantry = get_pantry_inventory()
    
    days_info = []
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for day in days:
        weather = get_weather_forecast(day)
        calendar = get_family_calendar(day)
        days_info.append({
            "day": day,
            "weather": weather,
            "headcount": calendar["headcount"],
            "events": calendar["events"],
            "recommended_dinner_time": calendar["recommended_dinner_time"],
            "prep_time_constraint": calendar["prep_time_constraint"]
        })
        
    preferences = {
        "dietary_constraints": profile.get("dietary_constraints", []),
        "target_macros": profile.get("target_macros", {}),
        "recipe_rotation": [r["name"] for r in profile.get("recipe_rotation", [])]
    }
    
    # Check if we have previous adjustments stored in workflow state
    adjustments = ctx.state.get("adjustments")
    
    # If adjustments exist, retrieve the previously generated menu so the LLM
    # can modify ONLY what was requested instead of generating from scratch.
    previous_menu = None
    if adjustments:
        current_menu_state = ctx.state.get("current_menu")
        if current_menu_state:
            # current_menu is stored as a WeeklyMenu dict or object by MenuGeneratorAgent
            if isinstance(current_menu_state, dict) and "meals" in current_menu_state:
                previous_menu = current_menu_state["meals"]
            elif hasattr(current_menu_state, "meals"):
                previous_menu = [m.model_dump() for m in current_menu_state.meals]
    
    return WeeklyContext(
        days_info=days_info,
        preferences=preferences,
        pantry=pantry,
        adjustments=adjustments,
        previous_menu=previous_menu
    )

# Prompt for Menu Generator Agent
menu_generator_instruction = """
You are Meal Concierge Multi-Agent, the master family meal planner. Your job is to generate a structured 7-day meal plan (from Monday to Sunday) based on:
1. Daily schedule and weather constraints (e.g., if sports matches are scheduled, suggest easy meals; if weather is rainy, disable outdoor grilling).
2. Daily headcount (custody schedule). Adjust food quantities accordingly.
3. Family dietary restrictions, macro targets, and picky eater preferences. Avoid forbidden foods (like seafood or mushrooms).
4. Utilize a mix of the family's 'recipe_rotation' favorites and 1-2 'new' recipe ideas.
5. Strictly match the ingredient names/keys to the ones listed in the pantry inventory when possible (e.g. use 'chicken_breast_g', 'pasta_g', 'flour_g', 'tomato_sauce_g', etc.). Do not invent new names for existing pantry items.

CRITICAL RULE FOR ADJUSTMENTS:
If the input contains BOTH 'adjustments' AND 'previous_menu', you MUST:
- Start from the 'previous_menu' as the base plan.
- ONLY change the specific day(s) or meal(s) that the user explicitly mentioned in 'adjustments'.
- Keep ALL other days exactly as they appear in 'previous_menu'. Do not regenerate them.
- Example: if adjustments says 'Change Wednesday to pasta', ONLY replace Wednesday's meal with a pasta dish. Monday, Tuesday, Thursday, Friday, Saturday, Sunday must remain IDENTICAL to previous_menu.

If there is no 'previous_menu', generate a fresh 7-day plan from scratch using all the constraints above.
"""

menu_generator_agent = LlmAgent(
    name="MenuGeneratorAgent",
    model=model_instance,
    instruction=menu_generator_instruction,
    output_schema=WeeklyMenu,
    output_key="current_menu"
)

def validate_dietary_constraints(menu: WeeklyMenu) -> List[str]:
    """Programmatic safety guardrail: returns a list of constraint violations found in the menu."""
    violations = []
    forbidden_keywords = {
        "seafood": ["seafood", "fish", "shrimp", "salmon", "tuna", "crab", "lobster", "cod", "halibut", "clams", "mussels"],
        "mushrooms": ["mushroom", "mushrooms", "shiitake", "portobello"],
        "raw onions": ["raw onion", "raw onions"]
    }
    
    for meal in menu.meals:
        meal_name_lower = meal.meal_name.lower()
        # Check meal name
        for category, words in forbidden_keywords.items():
            for word in words:
                if word in meal_name_lower:
                    violations.append(f"Day {meal.day}: Meal name '{meal.meal_name}' contains forbidden ingredient '{word}' ({category})")
                    
        # Check ingredients list
        for ing in meal.ingredients:
            ing_name_lower = ing.name.lower()
            for category, words in forbidden_keywords.items():
                for word in words:
                    if word in ing_name_lower:
                        violations.append(f"Day {meal.day}: Ingredient '{ing.name}' is forbidden ({category})")
                        
    return violations

# ---------------------------------------------------------------------------
# Menu Visualizer — OpenAI Image Generation (DALL-E & ChatGPT Image)
# Generates one representative food image for the weekly menu.
# Runs between MenuGeneratorAgent and user_review_node.
# ---------------------------------------------------------------------------

def _generate_image_sync(api_key: str, prompt: str) -> bytes:
    """Call OpenAI Image API synchronously via urllib and return raw image bytes."""
    import json
    import urllib.request
    import urllib.error
    import base64

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment variables.")

    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Try models in order of availability and quality
    models = ["chatgpt-image-latest", "gpt-image-2", "gpt-image-1.5", "dall-e-3"]
    last_err = None

    for model in models:
        payload = {
            "model": model,
            "prompt": prompt,
            "n": 1,
            "size": "1024x1024"
        }
        logger.info(f"MenuVisualizer: Trying OpenAI model '{model}'...")
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                resp_data = json.loads(response.read().decode("utf-8"))
                img_url = resp_data["data"][0]["url"]
                logger.info(f"MenuVisualizer: Model '{model}' succeeded. Downloading image...")
                
                # Download the image bytes from the returned URL
                with urllib.request.urlopen(img_url, timeout=30) as img_response:
                    return img_response.read()
        except urllib.error.HTTPError as e:
            err_body = e.fp.read().decode("utf-8")
            logger.warning(f"MenuVisualizer: Model '{model}' failed ({e.code} {e.reason}): {err_body}")
            # If billing limit is hit, raise immediately so we don't try other models
            if "billing" in err_body.lower() or "limit" in err_body.lower():
                raise RuntimeError(f"OpenAI billing limit reached: {err_body}")
            last_err = e
        except Exception as e:
            logger.warning(f"MenuVisualizer: Model '{model}' failed with error: {e}")
            last_err = e

    if last_err:
        raise last_err
    raise RuntimeError("All OpenAI image models failed.")


@node
async def menu_visualizer(ctx: Context, node_input: WeeklyMenu) -> Event:
    """
    Generates a visual food image for the weekly menu using OpenAI Image Generation.

    - Builds a descriptive prompt from the 7 meal names
    - Calls OpenAI Image API (isolated from Gemini quota pool)
    - Yields the image inline in the Playground chat
    - Passes WeeklyMenu through to user_review_node unchanged
    - Gracefully skips if OpenAI Image API is unavailable or OPENAI_API_KEY is missing
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    meal_names = [m.meal_name for m in node_input.meals]

    # Build a visually rich image prompt describing the week's meals
    meals_list = ", ".join(f"{node_input.meals[i].day}: {meal_names[i]}" for i in range(len(meal_names)))
    prompt = (
        f"A beautiful, professional food photography collage showing a weekly family meal plan. "
        f"The meals are: {meals_list}. "
        f"Warm, inviting kitchen aesthetic. Bright natural lighting. "
        f"High resolution, appetizing, magazine-quality food photography. "
        f"Flat-lay arrangement with colorful fresh ingredients visible."
    )

    try:
        logger.info("MenuVisualizer: Generating weekly menu image with OpenAI...")
        image_bytes = await asyncio.to_thread(_generate_image_sync, api_key, prompt)

        # Encode and display the image inline in the Playground
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")
        image_part = types.Part(
            inline_data=types.Blob(mime_type="image/png", data=img_b64)
        )
        caption_part = types.Part.from_text(
            text="🍽️ **Weekly Menu Visual** — Here's a preview of your proposed meals for the week:"
        )
        yield Event(
            content=types.Content(
                role="model",
                parts=[caption_part, image_part],
            )
        )
        logger.info("MenuVisualizer: Image generated and displayed successfully.")

    except Exception as e:
        # Graceful fallback — skip image, log warning, continue workflow
        logger.warning(f"MenuVisualizer: DALL-E 3 failed ({e}). Skipping image, continuing workflow.")
        yield Event(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(
                    text="🍽️ *(Menu image unavailable — continuing with text menu below)*"
                )],
            )
        )

    # Pass the WeeklyMenu through to user_review_node unchanged
    yield Event(output=node_input)


@node(rerun_on_resume=True)
async def user_review_node(ctx: Context, node_input: WeeklyMenu) -> Event:
    """Prompts the user for approval or edits of the generated weekly meal plan (Human-In-The-Loop)."""
    # Get current iteration count from workflow state to ensure unique interrupt_id per iteration
    review_count = ctx.state.get("review_count", 0)
    interrupt_id = f"user_approval_{review_count}"
    
    # Check if user has already responded for this iteration
    if not ctx.resume_inputs or interrupt_id not in ctx.resume_inputs:
        # Run programmatic safety guardrails
        violations = validate_dietary_constraints(node_input)
        
        # Display the proposed menu clearly to the user
        menu_text = "\n### Proposed Weekly Menu:\n"
        
        if violations:
            menu_text += "\n⚠️ **[SAFETY WARNING] Programmatic Guardrail Violations Found:**\n"
            for v in violations:
                menu_text += f"- *Violation*: {v}\n"
            menu_text += "\n"
            
        for meal in node_input.meals:
            menu_text += (
                f"- **{meal.day}** (Headcount: {get_family_calendar(meal.day)['headcount']}): "
                f"{meal.meal_name} ({meal.cooking_style}, Prep: {meal.prep_time_minutes}m, "
                f"Protein: {meal.estimated_protein_g}g, Cal: {meal.estimated_calories} kcal)\n"
                f"  * Ingredients: {', '.join(f'{ing.name} ({ing.amount}g/ml/units)' for ing in meal.ingredients)}\n"
            )
        menu_text += f"\n**Rationale**: {node_input.rationale}\n"
        
        # Output visual text to the console
        yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=menu_text)]))
        
        # Prompt for user approval
        yield RequestInput(
            interrupt_id=interrupt_id,
            message="Do you approve this weekly meal plan? Reply 'yes' to approve, or describe adjustments (e.g. 'Change Wednesday to pasta')."
        )
        return

    # User response is received
    raw_response = ctx.resume_inputs[interrupt_id]
    if isinstance(raw_response, dict):
        response = list(raw_response.values())[0].strip()
    else:
        response = str(raw_response).strip()
    
    if response.lower() in ["yes", "y", "approve", "approved"]:
        # User approved the menu
        yield Event(
            output=node_input,
            route="approved",
            state={"approved_menu": node_input.model_dump()}
        )
    else:
        # User requested changes, loop back to collect_context with the adjustment text string.
        # IMPORTANT: output must be a str here because collect_context expects node_input: Any (not WeeklyMenu).
        yield Event(
            output=response,
            route="adjust",
            state={"adjustments": response, "review_count": review_count + 1}
        )

@node
def grocery_list_builder(node_input: WeeklyMenu) -> Event:
    """Takes the approved menu, subtracts current pantry stock, and outputs a categorized shopping list."""
    pantry = get_pantry_inventory()
    required_ingredients = {}
    
    # 1. Sum up all required ingredients
    for meal in node_input.meals:
        for ing in meal.ingredients:
            required_ingredients[ing.name] = required_ingredients.get(ing.name, 0) + ing.amount
            
    # 2. Subtract pantry inventory
    shopping_list = {}
    for ing, req_qty in required_ingredients.items():
        pantry_qty = pantry.get(ing, 0)
        needed = req_qty - pantry_qty
        if needed > 0:
            shopping_list[ing] = needed

    # 3. Format categorized shopping list
    output_text = "\n### 🛒 Approved! Your Categorized Grocery List:\n"
    if not shopping_list:
        output_text += "You already have all the ingredients in your pantry!\n"
    else:
        for ing, qty in shopping_list.items():
            # Basic categorization based on suffix/keywords
            category = "Produce / Fresh"
            if "cheese" in ing or "cream" in ing or "butter" in ing:
                category = "Dairy"
            elif "beef" in ing or "chicken" in ing or "patties" in ing:
                category = "Meat"
            elif "pasta" in ing or "flour" in ing or "bean" in ing or "sauce" in ing:
                category = "Pantry Staples"
            
            output_text += f"- [{category}] {ing.replace('_g','').replace('_ml','').replace('_',' ')}: {qty} units/g/ml\n"
            
    # Yield content for the user interface
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=output_text)]))
    yield Event(output={"shopping_list": shopping_list, "menu": node_input.model_dump()})

# Prompt for Feedback Processor
feedback_instruction = """
You are Meal Concierge Multi-Agent's memory assistant. The user wants to rate a meal or provide feedback.
Your job is to:
1. Parse the user's query to identify which meal they are referring to.
2. Determine if the sentiment is positive (e.g. 'I loved it', 'it was great') or negative.
3. If positive, construct a recipe structure (using mock/reasonable ingredients matching the meal name) and output it so it can be saved.
4. Output a friendly response confirming that their preferences have been updated.
"""

class FeedbackOutput(BaseModel):
    meal_name: str = Field(description="Name of the meal being evaluated")
    sentiment: str = Field(description="Either 'positive' or 'negative'")
    recipe_details: Optional[DailyMeal] = Field(default=None, description="Recipe details if the sentiment is positive and should be saved to favorites")
    message: str = Field(description="Polite confirmation message to show to the user")

feedback_agent = LlmAgent(
    name="FeedbackAgent",
    model=model_instance,
    instruction=feedback_instruction,
    output_schema=FeedbackOutput
)

@node
def feedback_processor(node_input: FeedbackOutput) -> str:
    """Takes parsed feedback, updates the recipe rotation in long-term memory, and displays confirmation."""
    if node_input.sentiment == "positive" and node_input.recipe_details:
        details = node_input.recipe_details
        # Transform List[IngredientItem] to Dict[str, int] for user_profile.json
        ing_dict = {}
        if details.ingredients:
            for ing in details.ingredients:
                ing_dict[ing.name] = ing.amount
                
        transformed_recipe = {
            "name": details.meal_name,
            "ingredients": ing_dict,
            "prep_time_minutes": details.prep_time_minutes,
            "cook_time_minutes": 15,  # Mock default cook time
            "cooking_style": details.cooking_style,
            "protein_g": details.estimated_protein_g,
            "fat_g": details.estimated_fat_g,
            "calories": details.estimated_calories
        }
        added = add_recipe_to_rotation(transformed_recipe)
        if added:
            confirm = f"\n[Memory Update] Added '{node_input.meal_name}' to your family favorites rotation.\n"
        else:
            confirm = f"\n[Memory Update] '{node_input.meal_name}' is already in your family favorites.\n"
    elif node_input.sentiment == "negative":
        removed = remove_recipe_from_rotation(node_input.meal_name)
        if removed:
            confirm = f"\n[Memory Update] Removed '{node_input.meal_name}' from your family favorites rotation due to negative feedback.\n"
        else:
            confirm = f"\n[Memory Update] Feedback recorded: Negative sentiment for '{node_input.meal_name}' (Not in favorites list).\n"
    else:
        confirm = f"\n[Memory Update] Feedback recorded for '{node_input.meal_name}'.\n"
        
    result_message = f"{node_input.message}\n{confirm}"
    yield Event(content=types.Content(role="model", parts=[types.Part.from_text(text=result_message)]))
    yield Event(output=result_message)

# ==========================================
# 3. Connect Workflow Graph
# ==========================================

chef_workflow = Workflow(
    name="MealConciergeMultiAgentWorkflow",
    edges=[
        # Step 1: Embedding Semantic Router — no LLM call, uses cosine similarity
        # Model: gemini-embedding-1 (RPD: 1000, isolated quota pool)
        (START, embedding_intent_router),

        # Step 2: Route to correct branch based on embedding similarity score
        (embedding_intent_router, {"plan": collect_context, "feedback": feedback_agent}),

        # Plan flow: collect context → generate menu → visualize → user review (HITL)
        (collect_context, menu_generator_agent),
        (menu_generator_agent, menu_visualizer),    # 🆕 DALL-E 3 step
        (menu_visualizer, user_review_node),

        # Human-in-the-loop review routing
        (user_review_node, {"adjust": collect_context, "approved": grocery_list_builder}),

        # Feedback flow: feedback agent → memory update
        (feedback_agent, feedback_processor)
    ]
)

root_agent = chef_workflow

app = App(
    root_agent=root_agent,
    name="app",
    resumability_config=ResumabilityConfig(enabled=True)
)
