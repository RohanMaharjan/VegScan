import os
import requests
from django.conf import settings

SPOON_API_KEY = getattr(settings, 'SPOONACULAR_API_KEY', os.environ.get('SPOONACULAR_API_KEY'))
BASE = 'https://api.spoonacular.com'


def search_recipes_by_ingredients(ingredients, number=6):
    """Search recipes by list of ingredients using Spoonacular. Returns list of recipe dicts.
    Only uses scanned ingredients - minimizes recipes with extra/missing ingredients.
    If API key is not set, returns empty list to let caller fallback to local recipe generation.
    """
    if not SPOON_API_KEY:
        return []
    url = f"{BASE}/recipes/findByIngredients"
    params = {
        'apiKey': SPOON_API_KEY,
        'ingredients': ','.join(ingredients),
        'number': number,
        'ranking': 2,  # minimize missing ingredients = prioritize recipes using ONLY scanned vegs
        'ignorePantry': False  # allow basic cooking ingredients (oil, salt, spices)
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        return []
    return resp.json()


def get_recipe_information(recipe_id):
    """Get recipe information including instructions. Returns dict or None."""
    if not SPOON_API_KEY:
        return None
    url = f"{BASE}/recipes/{recipe_id}/information"
    params = {'apiKey': SPOON_API_KEY, 'includeNutrition': False}
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        return None
    return resp.json()
