import logging
import os
from functools import wraps
from pprint import pprint

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from Helpers import is_user_allowed

load_dotenv()

# Telegram and OpenRouter credentials and defaults (from os.environ after dotenv).
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = os.getenv(
    "OPENROUTER_BASE_URL",
    "https://openrouter.ai/api/v1",
).rstrip("/")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash")

ALLOWED_TELEGRAM_IDS = frozenset(
    int(part.strip())
    for part in os.environ["ALLOWED_TELEGRAM_IDS"].split(",")
    if part.strip()
)


def allowlisted(handler):
    """Wrap a handler so it runs only for Telegram user ids in ALLOWED_TELEGRAM_IDS."""

    @wraps(handler)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_user_allowed(update, ALLOWED_TELEGRAM_IDS):
            return
        await handler(update, context)

    return wrapped

@allowlisted
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start the bot, create user profile if needed, show main actions (Commands.txt: /start)."""
    print("[CMD] /start", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Development stub for listing help text; product flow uses /mainmenu and inline buttons."""
    print("[CMD] /help", f"args={context.args!r}")
    pprint(update.to_dict())


# Image / suggestion


@allowlisted
async def handle_new_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start a new image-based meal suggestion session; user can send multiple images (Commands.txt: /newsession)."""
    print("[CMD] /new_session", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_add_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Signal that the user finished sending images so the batch can be processed (Commands.txt: /done; registered as add_images)."""
    print("[CMD] /add_images", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_suggest_from_detection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate exactly three meal suggestions from the finalized ingredient list (Commands.txt: /suggest)."""
    print("[CMD] /suggest_from_detection", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_suggest_from_inventory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate exactly three suggestions from saved inventory only, without new images (Commands.txt: /inventorysuggest)."""
    print("[CMD] /suggest_from_inventory", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_choose_meal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Select one of the three suggested meals (Commands.txt: /choose1, /choose2, /choose3)."""
    print("[CMD] /choose_meal", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_how_to_cook(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send detailed How to Cook instructions for the chosen meal (Commands.txt: /cook)."""
    print("[CMD] /how_to_cook", f"args={context.args!r}")
    pprint(update.to_dict())


# Manual correction


@allowlisted
async def handle_add_ingredient(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Add a missing ingredient to the current session ingredient list (Commands.txt: /addingredient)."""
    print("[CMD] /add_ingredient", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_remove_ingredient(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Remove a wrongly detected ingredient from the current session list (Commands.txt: /removeingredient)."""
    print("[CMD] /remove_ingredient", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_update_inventory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Manually update saved inventory by adding or removing ingredients (Commands.txt: /updateinventory)."""
    print("[CMD] /update_inventory", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_log_meal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log a meal manually without image detection (Commands.txt: /loginmeal)."""
    print("[CMD] /log_meal", f"args={context.args!r}")
    pprint(update.to_dict())


# Diet


@allowlisted
async def handle_set_calorie_goal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set the user's daily calorie goal (Commands.txt: /setcalories)."""
    print("[CMD] /set_calorie_goal", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_set_meals_per_day(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Set how many meals per day the user wants (Commands.txt: /setmeals)."""
    print("[CMD] /set_meals_per_day", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_vegan_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enable vegan mode for future suggestions (Commands.txt: /veganon)."""
    print("[CMD] /vegan_on", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_vegan_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Disable vegan mode back to default non-vegan behavior (Commands.txt: /veganoff)."""
    print("[CMD] /vegan_off", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_diet_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reset diet to defaults: no calorie goal, no meal count rule, vegan off (Commands.txt: /resetdiet)."""
    print("[CMD] /diet_reset", f"args={context.args!r}")
    pprint(update.to_dict())


# Analytics


@allowlisted
async def handle_avg_calories(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show average calories per day for a selected period (Commands.txt: /avgcalories)."""
    print("[CMD] /avg_calories", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_calories_graph(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Produce a calories-per-day graph (e.g. last 7 or 30 days) (Commands.txt: /caloriegraph)."""
    print("[CMD] /calories_graph", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_meals_per_day(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show how many meals were logged per day (Commands.txt: /mealsperday)."""
    print("[CMD] /meals_per_day", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_goal_adherence(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show days under, near, or over calorie goal (Commands.txt: /goaladherence)."""
    print("[CMD] /goal_adherence", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_meal_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show recent meals summary: names, calories, dates (Commands.txt: /mealsummary)."""
    print("[CMD] /meal_history", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_most_used_ingredients(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show most-used ingredients from meal history (Commands.txt: /mostused)."""
    print("[CMD] /most_used_ingredients", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_detected_unused(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show ingredients seen often in detection or storage but rarely used in meals (Commands.txt: /unusedingredients)."""
    print("[CMD] /detected_unused", f"args={context.args!r}")
    pprint(update.to_dict())


@allowlisted
async def handle_meal_variety(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show meal variety and repetition patterns (Commands.txt: /variety)."""
    print("[CMD] /meal_variety", f"args={context.args!r}")
    pprint(update.to_dict())


# Non-command messages


@allowlisted
async def handle_non_command_message(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Plain text or photos (e.g. fridge images during a session); not a slash command in Commands.txt."""
    print("message received")
    pprint(update.to_dict())


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO
    )
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("help", handle_help))

    application.add_handler(CommandHandler("new_session", handle_new_session))
    application.add_handler(CommandHandler("add_images", handle_add_images))
    application.add_handler(CommandHandler("suggest_from_detection", handle_suggest_from_detection))
    application.add_handler(
        CommandHandler("suggest_from_inventory", handle_suggest_from_inventory)
    )
    application.add_handler(CommandHandler("choose_meal", handle_choose_meal))
    application.add_handler(CommandHandler("how_to_cook", handle_how_to_cook))

    application.add_handler(CommandHandler("add_ingredient", handle_add_ingredient))
    application.add_handler(CommandHandler("remove_ingredient", handle_remove_ingredient))
    application.add_handler(CommandHandler("update_inventory", handle_update_inventory))
    application.add_handler(CommandHandler("log_meal", handle_log_meal))

    application.add_handler(CommandHandler("set_calorie_goal", handle_set_calorie_goal))
    application.add_handler(CommandHandler("set_meals_per_day", handle_set_meals_per_day))
    application.add_handler(CommandHandler("vegan_on", handle_vegan_on))
    application.add_handler(CommandHandler("vegan_off", handle_vegan_off))
    application.add_handler(CommandHandler("diet_reset", handle_diet_reset))

    application.add_handler(CommandHandler("avg_calories", handle_avg_calories))
    application.add_handler(CommandHandler("calories_graph", handle_calories_graph))
    application.add_handler(CommandHandler("meals_per_day", handle_meals_per_day))
    application.add_handler(CommandHandler("goal_adherence", handle_goal_adherence))
    application.add_handler(CommandHandler("meal_history", handle_meal_history))
    application.add_handler(CommandHandler("most_used_ingredients", handle_most_used_ingredients))
    application.add_handler(CommandHandler("detected_unused", handle_detected_unused))
    application.add_handler(CommandHandler("meal_variety", handle_meal_variety))

    application.add_handler(
        MessageHandler(
            (filters.TEXT & ~filters.COMMAND) | filters.PHOTO,
            handle_non_command_message,
        )
    )

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
