from django.shortcuts import render
from django.http import HttpResponse
from . import api_client
from django.views.decorators.cache import cache_page
import random
import re


def _get_recipe_image(recipe_title, api_image=None, vegs=None):
    """
    Get the best recipe image from available sources.
    Priority: Spoonacular API image > generated food image based on vegetables
    """
    # Try Spoonacular image first
    if api_image:
        return api_image
    
    # Fallback: Generate image URL based on vegetables for better variety
    if vegs:
        # Use vegetables to create a unique, specific search query
        veg_list = '+'.join(v.replace('_', ' ').lower() for v in vegs)
        # Add different cooking methods for variety
        cooking_methods = ['curry', 'fry', 'stir fry', 'roast', 'saute', 'bake', 'grill']
        method = random.choice(cooking_methods)
        query = f"{veg_list}+{method}"
    else:
        # Fallback to recipe title if no vegs provided
        safe_title = recipe_title.replace(' ', '+')
        query = safe_title
    
    # Use Unsplash API to get food images (free, no auth required)
    return f"https://source.unsplash.com/500x500/?{query},food"


def _parse_amount_unit(text):
    """Try to parse an amount and unit from a string like '200g' or '2 tbsp'.
    Returns (amount: float or None, unit: str or None, raw: str)
    """
    if not text:
        return None, None, text
    text = str(text).strip()
    # match patterns like '2', '2.5', '200g', '2 tbsp', 'to taste'
    m = re.match(r'^([0-9]+(?:\.[0-9]+)?)(?:\s*)([a-zA-Z%]+)?', text)
    if m:
        amt = float(m.group(1))
        unit = m.group(2) or ''
        return amt, unit, text
    return None, None, text


def _standardize_ingredients_from_api(info):
    out = []
    for ing in info.get('extendedIngredients', []):
        name = ing.get('name')
        amt = ing.get('amount')
        unit = ing.get('unit') or ''
        try:
            amt_f = float(amt) if amt is not None else None
        except Exception:
            amt_f = None
        out.append({'name': name, 'amount': amt_f, 'unit': unit, 'raw': ing.get('originalString') or ''})
    return out


def _standardize_ingredients_from_fallback(list_of_dicts):
    out = []
    for item in list_of_dicts:
        name = item.get('name') if isinstance(item, dict) else str(item)
        amt_text = item.get('amount') if isinstance(item, dict) else None
        amt, unit, raw = _parse_amount_unit(amt_text)
        out.append({'name': name, 'amount': amt, 'unit': unit or '', 'raw': raw or ''})
    return out


def _parse_instructions_to_steps(instr_html):
    """
    Parse HTML instructions from API into a list of numbered steps.
    Handles <li>, <ol>, <p> tags and converts to clean step list.
    """
    if not instr_html:
        return []
    
    # Replace <li> with a marker to split on
    instr_html = instr_html.replace('<li>', '|||STEP|||')
    instr_html = instr_html.replace('</li>', '')
    instr_html = instr_html.replace('<ol>', '')
    instr_html = instr_html.replace('</ol>', '')
    instr_html = instr_html.replace('<p>', '')
    instr_html = instr_html.replace('</p>', ' ')
    instr_html = instr_html.replace('<br>', ' ')
    instr_html = instr_html.replace('<br/>', ' ')
    
    # Split by marker and clean up
    steps = [step.strip() for step in instr_html.split('|||STEP|||') if step.strip()]
    
    return steps if steps else []


def _make_fallback_recipe(vegs):
    # Enhanced recipe generator when API isn't available
    # Only use scanned vegetables - NO extras added
    veg_names = ' & '.join([v.replace('_', ' ').title() for v in vegs])
    
    # Step-by-step instructions
    instructions = [
        f"Wash all your {veg_names.lower()} thoroughly under running water.",
        "Peel and chop them into bite-sized pieces.",
        "Heat 2-3 tablespoons of oil in a large pan over medium heat.",
        "Add minced garlic and let it become fragrant (about 1 minute).",
        "Add harder vegetables first (like potatoes or carrots), cook for 3-4 minutes.",
        "Then add the remaining vegetables and stir-fry for another 4-6 minutes until everything is tender-crisp.",
        "Season with salt and pepper to your taste.",
        "Finish with a squeeze of lemon or lime if you like.",
        "Serve hot with steamed rice, flatbread, or as a side dish."
    ]
    
    ingredients = [{'name': v.replace('_', ' ').title(), 'amount': 200, 'unit': 'g'} for v in vegs]
    pantry = [
        {'name': 'Vegetable Oil', 'amount': 2, 'unit': 'tbsp'},
        {'name': 'Garlic', 'amount': 2, 'unit': 'cloves'},
        {'name': 'Salt', 'amount': 1, 'unit': 'tsp'},
        {'name': 'Black Pepper', 'amount': 0.5, 'unit': 'tsp'},
        {'name': 'Lemon (optional)', 'amount': 0.5, 'unit': 'fruit'}
    ]
    
    all_ings = _standardize_ingredients_from_fallback(pantry + ingredients)
    
    return {
        'title': f'Seasonal {veg_names} Stir-Fry',
        'image': _get_recipe_image(f'Seasonal {veg_names} Stir-Fry', vegs=vegs),
        'ingredients': all_ings,
        'instructions': instructions,  # Now a list of steps
        'servings': 2,
        'prepTime': 10,
        'cookTime': 15,
        'totalTime': 25,
        'diets': ['vegan', 'vegetarian'],
        'cuisines': ['asian'],
        'healthLabels': ['vegan', 'vegetarian', 'paleo-friendly', 'high-fiber'],
        'calories': []
    }


@cache_page(30)
def recipe_view(request):
    # Expected query params: veg=Bean&veg=Tomato ... model=<model>
    vegs = request.GET.getlist('veg')
    vegs = [v.replace('_',' ').strip() for v in vegs if v]

    # Always allow pantry spices
    pantry_allowed = ['salt', 'vegetable oil', 'black pepper', 'garlic']

    context = {'scanned': vegs, 'recipes': [], 'api_used': False}

    if not vegs:
        return render(request, 'recipes/recipe.html', context)

    # Try external API
    try:
        hits = api_client.search_recipes_by_ingredients(vegs, number=6)
        if hits:
            context['api_used'] = True
            recipes = []
            for h in hits:
                info = api_client.get_recipe_information(h.get('id')) or {}
                ingredients = _standardize_ingredients_from_api(info) if info else []
                
                # Extract rich metadata from API
                prep_time = info.get('preparationMinutes', 0) or 0
                cook_time = info.get('cookingMinutes', 0) or 0
                total_time = info.get('readyInMinutes', 0) or (prep_time + cook_time) or 30
                servings = info.get('servings', 2)
                
                # Parse instructions into step list
                instr = info.get('instructions') or ''
                instructions_list = _parse_instructions_to_steps(instr)
                
                # Extract diet labels and health labels if available
                diets = info.get('diets', [])
                labels = info.get('cuisines', [])
                health = info.get('healthLabels', [])[:5]  # limit to 5
                
                recipes.append({
                    'title': h.get('title'),
                    'image': _get_recipe_image(h.get('title'), h.get('image') or info.get('image'), vegs=vegs),
                    'usedIngredientCount': h.get('usedIngredientCount', 0),
                    'missedIngredientCount': h.get('missedIngredientCount', 0),
                    'ingredients': ingredients,
                    'instructions': instructions_list,
                    'sourceUrl': info.get('sourceUrl') if info else None,
                    'servings': servings,
                    'prepTime': prep_time,
                    'cookTime': cook_time,
                    'totalTime': total_time,
                    'diets': diets,
                    'cuisines': labels,
                    'healthLabels': health,
                    'calories': info.get('nutrition', {}).get('nutrients', []) if info.get('nutrition') else []
                })
            context['recipes'] = recipes
            return render(request, 'recipes/recipe.html', context)
    except Exception:
        # fallback handled below
        pass

    # Fallback local recipe (only using scanned vegetables - no extras)
    recipes_list = [_make_fallback_recipe(vegs)]
    context['recipes'] = recipes_list
    return render(request, 'recipes/recipe.html', context)
