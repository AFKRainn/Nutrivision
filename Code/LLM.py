"""OpenRouter LLM calls for meal generation and cooking guidance."""
from __future__ import annotations

import json
import os
import re

import httpx

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-v4-0324")


async def _chat(messages: list[dict]) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            json={"model": OPENROUTER_MODEL, "messages": messages},
        )
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    return match.group(1).strip() if match else text.strip()


async def generate_meal_suggestions(
    ingredients: list[str],
    calorie_goal: int | None,
    vegan: bool,
    recent_meals: list[str],
) -> list[dict]:
    """Return exactly 3 meal dicts: {name, description, calories, steps}."""
    diet_notes: list[str] = []
    if vegan:
        diet_notes.append("All meals must be fully vegan.")
    if calorie_goal:
        diet_notes.append(f"Each meal should be around {calorie_goal} calories.")
    if recent_meals:
        diet_notes.append(f"Do not suggest these recent meals: {', '.join(recent_meals)}.")

    prompt = (
        f"You are a cooking assistant. Suggest exactly 3 meals using these ingredients: "
        f"{', '.join(ingredients)}.\n"
        + (f"Diet rules: {' '.join(diet_notes)}\n" if diet_notes else "")
        + """
Respond ONLY with valid JSON, no extra text:
{
  "meals": [
    {
      "name": "Meal Name",
      "description": "One sentence description",
      "calories": 450,
      "steps": ["Step 1", "Step 2", "Step 3"]
    }
  ]
}
Give exactly 3 meals."""
    )

    text = await _chat([{"role": "user", "content": prompt}])
    data = json.loads(_extract_json(text))
    return data.get("meals", [])[:3]


async def generate_cooking_instructions(meal_name: str, ingredients: list[str]) -> str:
    """Return detailed step-by-step cooking instructions as plain text."""
    ing_note = f"Available ingredients: {', '.join(ingredients)}." if ingredients else ""
    prompt = (
        f"Give clear, detailed cooking instructions for: {meal_name}.\n"
        f"{ing_note}\n"
        "Format as numbered steps. Be practical and beginner-friendly. Plain text only."
    )
    return await _chat([{"role": "user", "content": prompt}])
