import asyncio
import logging
import os
import tempfile
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from Database import (
    init_db, ensure_user,
    get_diet, set_calorie_goal, set_meals_per_day, set_vegan, reset_diet,
    get_inventory, add_to_inventory, remove_from_inventory, clear_inventory,
    log_meal, get_meal_history,
    get_avg_calories, get_calories_per_day, get_meals_per_day,
    get_goal_adherence, get_meal_variety,
)
from Detector import DEFAULT_CONF, detect_image, merge_detections, weights_available
from Helpers import (
    is_user_allowed,
    format_ingredient_list,
    format_meal_suggestions,
    format_meal_history,
    format_inventory,
    format_diet,
    parse_int_arg,
)
from LLM import generate_meal_suggestions, generate_cooking_instructions

load_dotenv()

TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-0324")

ALLOWED_TELEGRAM_IDS = frozenset(
    int(p.strip()) for p in os.environ["ALLOWED_TELEGRAM_IDS"].split(",") if p.strip()
)


# ---------------------------------------------------------------------------
# Auth decorator
# ---------------------------------------------------------------------------

def allowlisted(handler):
    @wraps(handler)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not is_user_allowed(update, ALLOWED_TELEGRAM_IDS):
            return
        await handler(update, context)
    return wrapped


# ---------------------------------------------------------------------------
# Internal helpers (private to this file)
# ---------------------------------------------------------------------------

async def _download_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Path:
    photo = update.message.photo[-1]
    tg_file = await context.bot.get_file(photo.file_id)
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    await tg_file.download_to_drive(tmp.name)
    return Path(tmp.name)


def _run_detection_on_paths(image_paths: list[str]) -> tuple[dict[str, float], list[Path]]:
    all_detections: list[list[dict]] = []
    annotated_paths: list[Path] = []
    for raw in image_paths:
        dets, annotated = detect_image(Path(raw), conf=DEFAULT_CONF)
        all_detections.append(dets)
        if annotated:
            annotated_paths.append(annotated)
    return merge_detections(all_detections), annotated_paths


# ---------------------------------------------------------------------------
# /start  /help  /cancel  /mainmenu
# ---------------------------------------------------------------------------

@allowlisted
async def handle_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    await asyncio.to_thread(ensure_user, uid)
    model_note = "Model: ready." if weights_available() else "Model: best.pt not found — run train.py first."
    await update.message.reply_text(
        "Welcome to NutriVision!\n\n"
        "📸 /new_session — detect ingredients from fridge photos\n"
        "🥗 /inventorysuggest — suggest meals from saved inventory\n"
        "📦 /update_inventory show — manage your ingredient inventory\n"
        "⚖️ /show_diet — view or change diet settings\n"
        "📊 /avg_calories — view analytics\n\n"
        f"{model_note}"
    )


@allowlisted
async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_start(update, context)


@allowlisted
async def handle_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    await update.message.reply_text("Session cancelled. Use /start to begin again.")


@allowlisted
async def handle_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await handle_start(update, context)


# ---------------------------------------------------------------------------
# Image session: /new_session → send photos → /add_images → /suggest
# ---------------------------------------------------------------------------

@allowlisted
async def handle_new_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    context.user_data["session_active"] = True
    context.user_data["pending_images"] = []
    context.user_data["ingredients"] = {}

    if not weights_available():
        await update.message.reply_text(
            "Session started, but best.pt is missing. Detection will not work.\n"
            "Run train.py first."
        )
        return

    await update.message.reply_text(
        "New session started.\n"
        "Send one or more fridge / pantry photos.\n"
        "When finished, send /add_images or /done to run detection."
    )


@allowlisted
async def handle_add_images(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.user_data.get("session_active"):
        await update.message.reply_text("No active session. Start with /new_session.")
        return

    pending: list[str] = context.user_data.get("pending_images", [])
    if not pending:
        await update.message.reply_text("No photos received yet. Send images first.")
        return

    if not weights_available():
        await update.message.reply_text("best.pt not found. Run train.py first.")
        return

    await update.message.reply_text(f"Running detection on {len(pending)} image(s)…")

    try:
        ingredients, annotated_paths = await asyncio.to_thread(_run_detection_on_paths, pending)
    except Exception as exc:
        logging.exception("detection failed")
        await update.message.reply_text(f"Detection failed: {exc}")
        return

    context.user_data["ingredients"] = ingredients
    context.user_data["session_active"] = False

    await update.message.reply_text(
        format_ingredient_list(ingredients)
        + f"\n\n({len(pending)} image(s) processed)\n\n"
        "Fix mistakes: /add_ingredient  /remove_ingredient\n"
        "When ready: /suggest"
    )

    for ann_path in annotated_paths:
        with open(ann_path, "rb") as f:
            await update.message.reply_photo(f, caption="Detection overlay")


@allowlisted
async def handle_show_ingredients(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ingredients: dict[str, float] = context.user_data.get("ingredients", {})
    if not ingredients:
        await update.message.reply_text(
            "No ingredients yet. /new_session → send photos → /add_images."
        )
        return
    await update.message.reply_text(format_ingredient_list(ingredients))


# ---------------------------------------------------------------------------
# Manual ingredient correction (session list)
# ---------------------------------------------------------------------------

@allowlisted
async def handle_add_ingredient(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    ingredient = " ".join(args).strip().lower()
    if not ingredient:
        await update.message.reply_text("Usage: /add_ingredient <name>\nExample: /add_ingredient eggs")
        return

    ingredients: dict[str, float] = context.user_data.setdefault("ingredients", {})
    ingredients[ingredient] = 1.0
    await update.message.reply_text(
        f"Added '{ingredient}' to the ingredient list.\n\n"
        + format_ingredient_list(ingredients)
    )


@allowlisted
async def handle_remove_ingredient(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    ingredient = " ".join(args).strip().lower()
    if not ingredient:
        await update.message.reply_text("Usage: /remove_ingredient <name>\nExample: /remove_ingredient tomato")
        return

    ingredients: dict[str, float] = context.user_data.get("ingredients", {})
    if ingredient not in ingredients:
        await update.message.reply_text(f"'{ingredient}' not found in the list.")
        return

    del ingredients[ingredient]
    await update.message.reply_text(
        f"Removed '{ingredient}'.\n\n" + format_ingredient_list(ingredients)
    )


# ---------------------------------------------------------------------------
# Meal suggestion: /suggest  /inventorysuggest  /choose1-3  /cook
# ---------------------------------------------------------------------------

@allowlisted
async def handle_suggest_from_detection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    ingredients = context.user_data.get("ingredients", {})
    if not ingredients:
        await update.message.reply_text(
            "No ingredients detected. Start with /new_session and send photos."
        )
        return

    uid = update.effective_user.id
    diet = await asyncio.to_thread(get_diet, uid)
    recent = await asyncio.to_thread(get_meal_history, uid, 5)
    recent_names = [m["meal_name"] for m in recent]

    await update.message.reply_text("Generating meal suggestions…")

    try:
        meals = await generate_meal_suggestions(
            list(ingredients.keys()),
            diet["calorie_goal"],
            diet["vegan_mode"],
            recent_names,
        )
    except Exception as exc:
        logging.exception("LLM suggestion failed")
        await update.message.reply_text(f"Could not generate suggestions: {exc}")
        return

    if not meals:
        await update.message.reply_text("No suggestions returned. Try again.")
        return

    context.user_data["suggestions"] = meals
    await update.message.reply_text(format_meal_suggestions(meals), parse_mode="Markdown")


@allowlisted
async def handle_suggest_from_inventory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    inventory = await asyncio.to_thread(get_inventory, uid)

    if not inventory:
        await update.message.reply_text(
            "Your inventory is empty.\n"
            "Add ingredients: /update_inventory add milk eggs butter"
        )
        return

    diet = await asyncio.to_thread(get_diet, uid)
    recent = await asyncio.to_thread(get_meal_history, uid, 5)
    recent_names = [m["meal_name"] for m in recent]

    await update.message.reply_text("Generating meal suggestions from inventory…")

    try:
        meals = await generate_meal_suggestions(
            inventory,
            diet["calorie_goal"],
            diet["vegan_mode"],
            recent_names,
        )
    except Exception as exc:
        logging.exception("LLM suggestion failed")
        await update.message.reply_text(f"Could not generate suggestions: {exc}")
        return

    if not meals:
        await update.message.reply_text("No suggestions returned. Try again.")
        return

    context.user_data["suggestions"] = meals
    await update.message.reply_text(format_meal_suggestions(meals), parse_mode="Markdown")


@allowlisted
async def handle_choose_meal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Works for /choose1, /choose2, /choose3
    cmd = (update.message.text or "").split()[0].lstrip("/").lower()
    suggestions: list[dict] = context.user_data.get("suggestions", [])

    if not suggestions:
        await update.message.reply_text("No suggestions available. Use /suggest first.")
        return

    try:
        idx = int(cmd[-1]) - 1
        meal = suggestions[idx]
    except (ValueError, IndexError):
        await update.message.reply_text("Use /choose1, /choose2, or /choose3.")
        return

    context.user_data["chosen_meal"] = meal
    await update.message.reply_text(
        f"*{meal['name']}* selected!\n"
        f"{meal.get('description', '')}\n"
        f"~{meal.get('calories', '?')} kcal\n\n"
        "Send /cook for step-by-step instructions.",
        parse_mode="Markdown",
    )


@allowlisted
async def handle_how_to_cook(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    meal: dict | None = context.user_data.get("chosen_meal")
    if not meal:
        await update.message.reply_text("No meal chosen. Use /suggest then /choose1/2/3.")
        return

    ingredients = list(context.user_data.get("ingredients", {}).keys())
    uid = update.effective_user.id

    await update.message.reply_text(f"Getting instructions for {meal['name']}…")

    try:
        instructions = await generate_cooking_instructions(meal["name"], ingredients)
    except Exception as exc:
        logging.exception("LLM cooking instructions failed")
        await update.message.reply_text(f"Could not get instructions: {exc}")
        return

    await asyncio.to_thread(log_meal, uid, meal["name"], meal.get("calories", 0), "suggestion")

    await update.message.reply_text(
        f"*How to cook: {meal['name']}*\n\n{instructions}",
        parse_mode="Markdown",
    )


# ---------------------------------------------------------------------------
# Inventory management: /update_inventory
# ---------------------------------------------------------------------------

@allowlisted
async def handle_update_inventory(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    uid = update.effective_user.id

    if not args or args[0] not in ("add", "remove", "show", "clear"):
        await update.message.reply_text(
            "Inventory commands:\n"
            "/update_inventory show\n"
            "/update_inventory add milk eggs butter\n"
            "/update_inventory remove milk\n"
            "/update_inventory clear"
        )
        return

    action, items = args[0], [a.lower() for a in args[1:]]

    if action == "show":
        inv = await asyncio.to_thread(get_inventory, uid)
        await update.message.reply_text(format_inventory(inv))

    elif action == "add":
        if not items:
            await update.message.reply_text("Specify ingredients: /update_inventory add milk eggs")
            return
        for item in items:
            await asyncio.to_thread(add_to_inventory, uid, item)
        await update.message.reply_text(f"Added to inventory: {', '.join(items)}")

    elif action == "remove":
        if not items:
            await update.message.reply_text("Specify ingredient: /update_inventory remove milk")
            return
        for item in items:
            await asyncio.to_thread(remove_from_inventory, uid, item)
        await update.message.reply_text(f"Removed from inventory: {', '.join(items)}")

    elif action == "clear":
        await asyncio.to_thread(clear_inventory, uid)
        await update.message.reply_text("Inventory cleared.")


# ---------------------------------------------------------------------------
# Manual meal logging: /log_meal
# ---------------------------------------------------------------------------

@allowlisted
async def handle_log_meal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args or []
    if len(args) < 2:
        await update.message.reply_text(
            "Usage: /log_meal <meal name> <calories>\n"
            "Example: /log_meal Pasta 500"
        )
        return

    try:
        calories = int(args[-1])
        meal_name = " ".join(args[:-1])
    except ValueError:
        await update.message.reply_text("Last argument must be calories (number).\nExample: /log_meal Pasta 500")
        return

    uid = update.effective_user.id
    await asyncio.to_thread(log_meal, uid, meal_name, calories, "manual")
    await update.message.reply_text(f"Logged: {meal_name} ({calories} kcal)")


# ---------------------------------------------------------------------------
# Diet settings
# ---------------------------------------------------------------------------

@allowlisted
async def handle_set_calorie_goal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    goal = parse_int_arg(context.args or [])
    if not goal or goal <= 0:
        await update.message.reply_text("Usage: /set_calorie_goal <number>\nExample: /set_calorie_goal 2000")
        return
    await asyncio.to_thread(set_calorie_goal, update.effective_user.id, goal)
    await update.message.reply_text(f"Calorie goal set to {goal} kcal/day.")


@allowlisted
async def handle_set_meals_per_day(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    count = parse_int_arg(context.args or [])
    if not count or count <= 0:
        await update.message.reply_text("Usage: /set_meals_per_day <number>\nExample: /set_meals_per_day 3")
        return
    await asyncio.to_thread(set_meals_per_day, update.effective_user.id, count)
    await update.message.reply_text(f"Meals per day set to {count}.")


@allowlisted
async def handle_vegan_on(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await asyncio.to_thread(set_vegan, update.effective_user.id, True)
    await update.message.reply_text("Vegan mode enabled. Future suggestions will be fully vegan.")


@allowlisted
async def handle_vegan_off(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await asyncio.to_thread(set_vegan, update.effective_user.id, False)
    await update.message.reply_text("Vegan mode disabled.")


@allowlisted
async def handle_diet_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await asyncio.to_thread(reset_diet, update.effective_user.id)
    await update.message.reply_text("Diet settings reset to defaults.")


@allowlisted
async def handle_show_diet(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    diet = await asyncio.to_thread(get_diet, update.effective_user.id)
    await update.message.reply_text(format_diet(diet), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@allowlisted
async def handle_avg_calories(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    avg = await asyncio.to_thread(get_avg_calories, uid, 7)
    if avg is None:
        await update.message.reply_text("No meals logged yet.")
        return
    await update.message.reply_text(f"Average calories per day (last 7 days): *{avg:.0f} kcal*", parse_mode="Markdown")


@allowlisted
async def handle_calories_graph(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    rows = await asyncio.to_thread(get_calories_per_day, uid, 7)

    if not rows:
        await update.message.reply_text("No meal history to graph yet.")
        return

    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        days = [r["day"] for r in rows]
        cals = [r["total"] for r in rows]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(days, cals, color="#4CAF50")
        ax.set_title("Calories per day (last 7 days)")
        ax.set_ylabel("Calories (kcal)")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        plt.close()

        await update.message.reply_photo(buf, caption="Calories per day")

    except ImportError:
        lines = [f"• {r['day']}: {r['total']} kcal" for r in rows]
        await update.message.reply_text("Calories per day (last 7):\n" + "\n".join(lines))


@allowlisted
async def handle_meals_per_day(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    rows = await asyncio.to_thread(get_meals_per_day, uid, 7)
    if not rows:
        await update.message.reply_text("No meals logged yet.")
        return
    lines = ["*Meals per day (last 7 days):*"] + [f"• {r['day']}: {r['count']} meal(s)" for r in rows]
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


@allowlisted
async def handle_goal_adherence(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    result = await asyncio.to_thread(get_goal_adherence, uid, 7)
    if result.get("error") == "no_goal":
        await update.message.reply_text("No calorie goal set. Use /set_calorie_goal 2000 first.")
        return
    await update.message.reply_text(
        f"*Goal adherence (last 7 days)* — goal: {result['goal']} kcal\n"
        f"• Under goal: {result['under']} day(s)\n"
        f"• On target:  {result['near']} day(s)\n"
        f"• Over goal:  {result['over']} day(s)\n"
        f"• Days tracked: {result['days']}",
        parse_mode="Markdown",
    )


@allowlisted
async def handle_meal_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    history = await asyncio.to_thread(get_meal_history, uid, 10)
    await update.message.reply_text(format_meal_history(history), parse_mode="Markdown")


@allowlisted
async def handle_most_used_ingredients(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    inventory = await asyncio.to_thread(get_inventory, uid)
    if not inventory:
        await update.message.reply_text("Inventory is empty. Add items with /update_inventory add ...")
        return
    await update.message.reply_text(
        "*Your saved inventory (most likely frequently used):*\n"
        + "\n".join(f"• {i}" for i in inventory),
        parse_mode="Markdown",
    )


@allowlisted
async def handle_detected_unused(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    inventory = await asyncio.to_thread(get_inventory, uid)
    history = await asyncio.to_thread(get_meal_history, uid, 20)

    if not inventory:
        await update.message.reply_text("Inventory is empty.")
        return

    used_in_meals = " ".join(m["meal_name"].lower() for m in history)
    unused = [i for i in inventory if i not in used_in_meals]

    if not unused:
        await update.message.reply_text("All inventory ingredients appear in your meal history.")
        return

    await update.message.reply_text(
        "*Ingredients in inventory rarely seen in logged meals:*\n"
        + "\n".join(f"• {i}" for i in unused),
        parse_mode="Markdown",
    )


@allowlisted
async def handle_meal_variety(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    uid = update.effective_user.id
    result = await asyncio.to_thread(get_meal_variety, uid, 30)

    if result["total"] == 0:
        await update.message.reply_text("No meals logged in the last 30 days.")
        return

    lines = [
        f"*Meal variety (last 30 days):*",
        f"Total meals logged: {result['total']}",
        f"Unique meals: {result['unique']}\n",
    ]
    for m in result["breakdown"][:8]:
        lines.append(f"• {m['meal_name']} × {m['count']}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Non-command messages (photos during session)
# ---------------------------------------------------------------------------

@allowlisted
async def handle_non_command_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.photo:
        if not context.user_data.get("session_active"):
            await update.message.reply_text("Send /new_session first, then your photos.")
            return
        path = await _download_photo(update, context)
        pending: list[str] = context.user_data.setdefault("pending_images", [])
        pending.append(str(path))
        await update.message.reply_text(
            f"Photo {len(pending)} received. Send more or /add_images when done."
        )
        return

    if update.message.text:
        await update.message.reply_text("Use /new_session to start, then send photos. /start for help.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(message)s", level=logging.INFO)
    init_db()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Navigation
    application.add_handler(CommandHandler("start", handle_start))
    application.add_handler(CommandHandler("help", handle_help))
    application.add_handler(CommandHandler("cancel", handle_cancel))
    application.add_handler(CommandHandler("mainmenu", handle_main_menu))

    # Image session
    application.add_handler(CommandHandler("new_session", handle_new_session))
    application.add_handler(CommandHandler("newsession", handle_new_session))
    application.add_handler(CommandHandler("add_images", handle_add_images))
    application.add_handler(CommandHandler("done", handle_add_images))
    application.add_handler(CommandHandler("show_ingredients", handle_show_ingredients))
    application.add_handler(CommandHandler("showingredients", handle_show_ingredients))

    # Meal suggestions
    application.add_handler(CommandHandler("suggest", handle_suggest_from_detection))
    application.add_handler(CommandHandler("suggest_from_detection", handle_suggest_from_detection))
    application.add_handler(CommandHandler("inventorysuggest", handle_suggest_from_inventory))
    application.add_handler(CommandHandler("suggest_from_inventory", handle_suggest_from_inventory))
    application.add_handler(CommandHandler("choose1", handle_choose_meal))
    application.add_handler(CommandHandler("choose2", handle_choose_meal))
    application.add_handler(CommandHandler("choose3", handle_choose_meal))
    application.add_handler(CommandHandler("cook", handle_how_to_cook))
    application.add_handler(CommandHandler("how_to_cook", handle_how_to_cook))

    # Manual corrections
    application.add_handler(CommandHandler("add_ingredient", handle_add_ingredient))
    application.add_handler(CommandHandler("addingredient", handle_add_ingredient))
    application.add_handler(CommandHandler("remove_ingredient", handle_remove_ingredient))
    application.add_handler(CommandHandler("removeingredient", handle_remove_ingredient))

    # Inventory
    application.add_handler(CommandHandler("update_inventory", handle_update_inventory))
    application.add_handler(CommandHandler("updateinventory", handle_update_inventory))

    # Manual meal log
    application.add_handler(CommandHandler("log_meal", handle_log_meal))
    application.add_handler(CommandHandler("loginmeal", handle_log_meal))

    # Diet
    application.add_handler(CommandHandler("set_calorie_goal", handle_set_calorie_goal))
    application.add_handler(CommandHandler("setcalories", handle_set_calorie_goal))
    application.add_handler(CommandHandler("set_meals_per_day", handle_set_meals_per_day))
    application.add_handler(CommandHandler("setmeals", handle_set_meals_per_day))
    application.add_handler(CommandHandler("vegan_on", handle_vegan_on))
    application.add_handler(CommandHandler("veganon", handle_vegan_on))
    application.add_handler(CommandHandler("vegan_off", handle_vegan_off))
    application.add_handler(CommandHandler("veganoff", handle_vegan_off))
    application.add_handler(CommandHandler("diet_reset", handle_diet_reset))
    application.add_handler(CommandHandler("resetdiet", handle_diet_reset))
    application.add_handler(CommandHandler("show_diet", handle_show_diet))
    application.add_handler(CommandHandler("showdiet", handle_show_diet))

    # Analytics
    application.add_handler(CommandHandler("avg_calories", handle_avg_calories))
    application.add_handler(CommandHandler("avgcalories", handle_avg_calories))
    application.add_handler(CommandHandler("calories_graph", handle_calories_graph))
    application.add_handler(CommandHandler("caloriegraph", handle_calories_graph))
    application.add_handler(CommandHandler("meals_per_day", handle_meals_per_day))
    application.add_handler(CommandHandler("mealsperday", handle_meals_per_day))
    application.add_handler(CommandHandler("goal_adherence", handle_goal_adherence))
    application.add_handler(CommandHandler("goaladherence", handle_goal_adherence))
    application.add_handler(CommandHandler("meal_history", handle_meal_history))
    application.add_handler(CommandHandler("mealsummary", handle_meal_history))
    application.add_handler(CommandHandler("most_used_ingredients", handle_most_used_ingredients))
    application.add_handler(CommandHandler("mostused", handle_most_used_ingredients))
    application.add_handler(CommandHandler("detected_unused", handle_detected_unused))
    application.add_handler(CommandHandler("unusedingredients", handle_detected_unused))
    application.add_handler(CommandHandler("meal_variety", handle_meal_variety))
    application.add_handler(CommandHandler("variety", handle_meal_variety))

    application.add_handler(
        MessageHandler(
            (filters.TEXT & ~filters.COMMAND) | filters.PHOTO,
            handle_non_command_message,
        )
    )

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
