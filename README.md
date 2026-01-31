# NutriVision: Smart Cooking Assistant

NutriVision is an AI Telegram bot designed to simplify meal planning and nutrition tracking. The system helps users decide what to cook based on the ingredients they already have at home, while supporting their dietary goals and reducing food waste to increase personalization.

## How It Works

- **Visual Ingredient Recognition:** Uses a trained **YOLOv12** model to identify ingredients from photos of your fridge, pantry, or countertop.
- **Intelligent Recipe Suggestions:** Powered by **LLaMA** or **GPT**, the bot provides three personalized meal suggestions based on identified ingredients and user preferences.
- **Personalized Nutrition Tracking:** Supports dietary goals (calories, meal frequency, vegan/vegetarian options) and tracks nutritional intake using APIs like **OpenFoodFacts**.
- **Inventory Management:** Automatically updates your home inventory based on what you cook and helps you use ingredients before they expire.
- **Seamless Telegram Integration:** No need for a new app—interact with the assistant directly through a familiar messaging interface..


## work flow
1. **Capture:** Send photos of your ingredients to the Telegram bot.
2. **Identify:** The YOLOv12 model detects the items and creates a unified list.
3. **Refine:** Manually add or remove items to ensure the inventory is 100% accurate.
4. **Suggest:** Receive three meal options tailored to your diet and history.
5. **Cook:** Choose a meal and get step-by-step cooking instructions.
6. **Track:** The system logs your meal, calories, and updates your inventory.