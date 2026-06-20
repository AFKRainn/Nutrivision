"""Formatting and utility helpers for NutriVision."""
from __future__ import annotations

from telegram import Update


def is_user_allowed(update: Update, allowed_ids: frozenset[int]) -> bool:
    user = update.effective_user
    return user is not None and user.id in allowed_ids


def format_ingredient_list(ingredients: dict[str, float]) -> str:
    if not ingredients:
        return "No ingredients detected."
    lines = [f"• {name} ({conf:.0%})" for name, conf in ingredients.items()]
    return "Detected ingredients:\n" + "\n".join(lines)


def format_meal_suggestions(meals: list[dict]) -> str:
    if not meals:
        return "Could not generate meal suggestions."
    lines: list[str] = []
    for i, meal in enumerate(meals, 1):
        lines.append(f"*{i}. {meal['name']}*")
        lines.append(meal.get("description", ""))
        lines.append(f"~{meal.get('calories', '?')} kcal\n")
    lines.append("Choose: /choose1   /choose2   /choose3")
    return "\n".join(lines)


def format_diet_settings(diet: dict) -> str:
    goal = diet.get("calorie_goal")
    meals = diet.get("meals_per_day")
    vegan = diet.get("vegan_mode", False)
    return (
        "*Current diet settings:*\n"
        f"• Calorie goal: {goal} kcal/day\n" if goal else "• Calorie goal: not set\n"
        f"• Meals per day: {meals}\n" if meals else "• Meals per day: not set\n"
        f"• Vegan mode: {'on ✓' if vegan else 'off'}"
    )


def format_meal_history(history: list[dict]) -> str:
    if not history:
        return "No meals logged yet."
    lines = ["*Recent meals:*"]
    for m in history:
        lines.append(f"• {m['meal_name']} — {m['calories']} kcal  ({m['logged_at']})")
    return "\n".join(lines)


def format_inventory(inventory: list[str]) -> str:
    if not inventory:
        return "Your inventory is empty.\nAdd items: /update_inventory add milk eggs butter"
    return "Current inventory:\n" + "\n".join(f"• {i}" for i in sorted(inventory))


def format_diet(diet: dict) -> str:
    goal = diet.get("calorie_goal")
    meals = diet.get("meals_per_day")
    vegan = diet.get("vegan_mode", False)
    lines = ["*Diet settings:*"]
    lines.append(f"• Calorie goal: {goal} kcal/day" if goal else "• Calorie goal: not set")
    lines.append(f"• Meals per day: {meals}" if meals else "• Meals per day: not set")
    lines.append(f"• Vegan mode: {'on ✓' if vegan else 'off'}")
    return "\n".join(lines)


def parse_int_arg(args: list[str]) -> int | None:
    try:
        return int(args[0]) if args else None
    except ValueError:
        return None
