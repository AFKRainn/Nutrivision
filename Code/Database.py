"""SQLite persistence layer for NutriVision."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "nutrivision.db"


@contextmanager
def _cursor():
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        yield cur
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with _cursor() as cur:
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                telegram_id INTEGER PRIMARY KEY,
                calorie_goal INTEGER,
                meals_per_day INTEGER,
                vegan_mode INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS inventory (
                telegram_id INTEGER,
                ingredient TEXT,
                PRIMARY KEY (telegram_id, ingredient)
            );
            CREATE TABLE IF NOT EXISTS meal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                telegram_id INTEGER,
                meal_name TEXT,
                calories INTEGER,
                logged_at TEXT DEFAULT (date('now')),
                source TEXT DEFAULT 'suggestion'
            );
        """)


def ensure_user(telegram_id: int) -> None:
    with _cursor() as cur:
        cur.execute("INSERT OR IGNORE INTO users (telegram_id) VALUES (?)", (telegram_id,))


# --- Diet ---

def get_diet(telegram_id: int) -> dict:
    with _cursor() as cur:
        cur.execute(
            "SELECT calorie_goal, meals_per_day, vegan_mode FROM users WHERE telegram_id=?",
            (telegram_id,),
        )
        row = cur.fetchone()
    if row is None:
        return {"calorie_goal": None, "meals_per_day": None, "vegan_mode": False}
    return {
        "calorie_goal": row["calorie_goal"],
        "meals_per_day": row["meals_per_day"],
        "vegan_mode": bool(row["vegan_mode"]),
    }


def set_calorie_goal(telegram_id: int, goal: int | None) -> None:
    ensure_user(telegram_id)
    with _cursor() as cur:
        cur.execute("UPDATE users SET calorie_goal=? WHERE telegram_id=?", (goal, telegram_id))


def set_meals_per_day(telegram_id: int, count: int | None) -> None:
    ensure_user(telegram_id)
    with _cursor() as cur:
        cur.execute("UPDATE users SET meals_per_day=? WHERE telegram_id=?", (count, telegram_id))


def set_vegan(telegram_id: int, enabled: bool) -> None:
    ensure_user(telegram_id)
    with _cursor() as cur:
        cur.execute("UPDATE users SET vegan_mode=? WHERE telegram_id=?", (int(enabled), telegram_id))


def reset_diet(telegram_id: int) -> None:
    ensure_user(telegram_id)
    with _cursor() as cur:
        cur.execute(
            "UPDATE users SET calorie_goal=NULL, meals_per_day=NULL, vegan_mode=0 WHERE telegram_id=?",
            (telegram_id,),
        )


# --- Inventory ---

def get_inventory(telegram_id: int) -> list[str]:
    with _cursor() as cur:
        cur.execute(
            "SELECT ingredient FROM inventory WHERE telegram_id=? ORDER BY ingredient",
            (telegram_id,),
        )
        return [row["ingredient"] for row in cur.fetchall()]


def add_to_inventory(telegram_id: int, ingredient: str) -> None:
    ensure_user(telegram_id)
    with _cursor() as cur:
        cur.execute(
            "INSERT OR IGNORE INTO inventory (telegram_id, ingredient) VALUES (?, ?)",
            (telegram_id, ingredient.lower().strip()),
        )


def remove_from_inventory(telegram_id: int, ingredient: str) -> None:
    with _cursor() as cur:
        cur.execute(
            "DELETE FROM inventory WHERE telegram_id=? AND ingredient=?",
            (telegram_id, ingredient.lower().strip()),
        )


def clear_inventory(telegram_id: int) -> None:
    with _cursor() as cur:
        cur.execute("DELETE FROM inventory WHERE telegram_id=?", (telegram_id,))


# --- Meal history ---

def log_meal(telegram_id: int, meal_name: str, calories: int, source: str = "suggestion") -> None:
    ensure_user(telegram_id)
    with _cursor() as cur:
        cur.execute(
            "INSERT INTO meal_history (telegram_id, meal_name, calories, source) VALUES (?, ?, ?, ?)",
            (telegram_id, meal_name, calories, source),
        )


def get_meal_history(telegram_id: int, limit: int = 10) -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT meal_name, calories, logged_at, source FROM meal_history "
            "WHERE telegram_id=? ORDER BY logged_at DESC LIMIT ?",
            (telegram_id, limit),
        )
        return [dict(row) for row in cur.fetchall()]


# --- Analytics ---

def get_calories_per_day(telegram_id: int, days: int = 7) -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT date(logged_at) as day, SUM(calories) as total FROM meal_history "
            "WHERE telegram_id=? AND logged_at >= date('now', ?) GROUP BY day ORDER BY day",
            (telegram_id, f"-{days} days"),
        )
        return [dict(row) for row in cur.fetchall()]


def get_avg_calories(telegram_id: int, days: int = 7) -> float | None:
    rows = get_calories_per_day(telegram_id, days)
    if not rows:
        return None
    return sum(r["total"] for r in rows) / len(rows)


def get_meals_per_day(telegram_id: int, days: int = 7) -> list[dict]:
    with _cursor() as cur:
        cur.execute(
            "SELECT date(logged_at) as day, COUNT(*) as count FROM meal_history "
            "WHERE telegram_id=? AND logged_at >= date('now', ?) GROUP BY day ORDER BY day",
            (telegram_id, f"-{days} days"),
        )
        return [dict(row) for row in cur.fetchall()]


def get_goal_adherence(telegram_id: int, days: int = 7) -> dict:
    diet = get_diet(telegram_id)
    goal = diet.get("calorie_goal")
    if not goal:
        return {"error": "no_goal"}
    rows = get_calories_per_day(telegram_id, days)
    under = sum(1 for r in rows if r["total"] < goal * 0.9)
    near = sum(1 for r in rows if goal * 0.9 <= r["total"] <= goal * 1.1)
    over = sum(1 for r in rows if r["total"] > goal * 1.1)
    return {"goal": goal, "under": under, "near": near, "over": over, "days": len(rows)}


def get_meal_variety(telegram_id: int, days: int = 30) -> dict:
    with _cursor() as cur:
        cur.execute(
            "SELECT meal_name, COUNT(*) as count FROM meal_history "
            "WHERE telegram_id=? AND logged_at >= date('now', ?) "
            "GROUP BY meal_name ORDER BY count DESC",
            (telegram_id, f"-{days} days"),
        )
        rows = [dict(r) for r in cur.fetchall()]
    total = sum(r["count"] for r in rows)
    return {"total": total, "unique": len(rows), "breakdown": rows}
