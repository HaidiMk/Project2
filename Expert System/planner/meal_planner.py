import copy
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]         
sys.path.insert(0, str(BASE_DIR))

TOPSIS_DIR = Path(__file__).resolve().parents[2] / "TOPSIS"
if str(TOPSIS_DIR) not in sys.path:
    sys.path.insert(0, str(TOPSIS_DIR))

from core.constants import NUTRIENT_COLS             
from core.user_profile import UserProfile            
from engine.filtering_engine import DietaryExpertSystem  
from topsis_model import rank_with_topsis             

MEAL_SLOTS: Tuple[str, ...] = ("breakfast", "lunch", "dinner")

DEFAULT_TOLERANCE = 0.15
MAX_TOLERANCE = 0.40           
FALLBACK_TOLERANCE_STEP = 0.10 
DEFAULT_TOP_K_PER_BAND = 40
N_CALORIE_BANDS = 3

RANK_WEIGHT = 0.7
FIT_WEIGHT = 1.0 - RANK_WEIGHT

MEAL_OUTPUT_COLS: Tuple[str, ...] = (
    "RecipeId", "Name", *NUTRIENT_COLS, "final_score",
)
TOTAL_COLS: Tuple[str, ...] = (
    "Calories", "ProteinContent", "SugarContent", "FiberContent",
)


def _validate_profile(profile) -> None:
    if not isinstance(profile, UserProfile):
        raise TypeError(
            f"profile must be a UserProfile (core.user_profile), got "
            f"{type(profile)}")


def _validate_system(system) -> None:
    if system is not None and not isinstance(system, DietaryExpertSystem):
        raise TypeError(
            f"system must be a DietaryExpertSystem instance or None, got "
            f"{type(system)}")


def _validate_tolerance(tolerance: float) -> None:
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError(f"tolerance must be a number, got {tolerance!r}")
    if not (0 < tolerance <= MAX_TOLERANCE):
        raise ValueError(
            f"tolerance must be in (0, {MAX_TOLERANCE}], got {tolerance!r}")


def _validate_top_k(top_k_per_band: int) -> None:
    if not isinstance(top_k_per_band, int) or isinstance(top_k_per_band, bool):
        raise TypeError(f"top_k_per_band must be an int, got {top_k_per_band!r}")
    if top_k_per_band < 1:
        raise ValueError(
            f"top_k_per_band must be a positive int, got {top_k_per_band!r}")


def _validate_days(days: int) -> None:
    if not isinstance(days, int) or isinstance(days, bool):
        raise TypeError(f"days must be an int, got {days!r}")
    if not 1 <= days <= 31:
        raise ValueError(f"days must be in [1, 31], got {days!r}")


def _validate_exclude(exclude_recipe_ids) -> set:
    if exclude_recipe_ids is None:
        return set()
    return {int(rid) for rid in exclude_recipe_ids}



def _load_system() -> DietaryExpertSystem:
    from main import load_data
    return DietaryExpertSystem(load_data())


def _ensure_system(system) -> DietaryExpertSystem:
    _validate_system(system)
    return system if system is not None else _load_system()


def _slot_pipeline(system: DietaryExpertSystem, profile: UserProfile,
                   meal_type: str, user_taste_text: Optional[str]):
    _validate_profile(profile)
    slot_profile = copy.copy(profile)
    slot_profile.meal_type = meal_type
    result = system.filter_recipes(slot_profile)
    ranked = rank_with_topsis(result["safe_recipes"], profile.goal,
                              user_taste_text=user_taste_text)
    return result, ranked


def _stratified_pool(ranked: pd.DataFrame, top_k_per_band: int) -> pd.DataFrame:
    if len(ranked) == 0:
        return ranked
    with_cal = ranked.dropna(subset=["Calories"])
    if len(with_cal) <= N_CALORIE_BANDS * top_k_per_band:
        return with_cal.sort_values("final_score", ascending=False).reset_index(drop=True)
    cal = with_cal["Calories"].astype(float)
    lo, hi = float(cal.min()), float(cal.max())
    if hi <= lo:
        return with_cal.sort_values("final_score", ascending=False).reset_index(drop=True)
    parts = []
    eps = max(1e-6, (hi - lo) * 1e-6)
    for i in range(N_CALORIE_BANDS):
        b_lo = lo + (hi - lo) * i / N_CALORIE_BANDS
        b_hi = lo + (hi - lo) * (i + 1) / N_CALORIE_BANDS + (eps if i == N_CALORIE_BANDS - 1 else 0.0)
        parts.append(with_cal[(cal >= b_lo) & (cal < b_hi)].head(top_k_per_band))
    return pd.concat(parts).sort_values(
        "final_score", ascending=False).reset_index(drop=True)


def _excluded_pool(pool: pd.DataFrame, exclude_ids: set) -> Tuple[pd.DataFrame, bool]:
    if not exclude_ids:
        return pool, False
    used_mask = pool["RecipeId"].isin(exclude_ids)
    remaining = pool[~used_mask]
    if len(remaining) == 0:
        return pool, True
    return remaining, False


def _selection_score(mean_final, dev: float):
    return (RANK_WEIGHT * mean_final
            + FIT_WEIGHT * (1.0 - np.minimum(np.abs(dev), 1.0)))


def _combo_search(pool_b: pd.DataFrame, pool_l: pd.DataFrame,
                  pool_d: pd.DataFrame, target: float,
                  tolerance: Optional[float]):
    if len(pool_b) == 0 or len(pool_l) == 0 or len(pool_d) == 0:
        return None
    B, L, D = (pool_b.reset_index(drop=True), pool_l.reset_index(drop=True),
               pool_d.reset_index(drop=True))
    Bc = B["Calories"].astype(float).to_numpy()
    Lc = L["Calories"].astype(float).to_numpy()
    Dc = D["Calories"].astype(float).to_numpy()
    Bf = B["final_score"].astype(float).to_numpy()
    Lf = L["final_score"].astype(float).to_numpy()
    Df = D["final_score"].astype(float).to_numpy()
    Bids = B["RecipeId"].to_numpy()
    Lids = L["RecipeId"].to_numpy()
    Dids = D["RecipeId"].to_numpy()

    closest_mode = tolerance is None
    lo = None if closest_mode else target * (1 - tolerance)
    hi = None if closest_mode else target * (1 + tolerance)

    best = None  

    def _track(sel, mean_final, s, bidx, lidx, didx):
        nonlocal best
        if closest_mode:
            better = sel < best[0] if best is not None else True
        else:
            better = sel > best[0] if best is not None else True
        if better:
            best = (sel, mean_final, s, [bidx, lidx, didx])

    for di in range(len(Dc)):
        pair = Bc[:, None] + Lc[None, :]
        total = pair + Dc[di]
        if closest_mode:
            mask = np.ones(total.shape, dtype=bool)
        else:
            mask = (total >= lo) & (total <= hi)
        if not mask.any():
            continue
        mB, mL = np.where(mask)
        valid = ((Bids[mB] != Lids[mL]) & (Bids[mB] != Dids[di])
                 & (Lids[mL] != Dids[di]))
        if not valid.any():
            continue
        mB, mL = mB[valid], mL[valid]
        dev = total[mB, mL] / target - 1.0
        mean_final = (Bf[mB] + Lf[mL] + Df[di]) / 3.0

        if closest_mode:
            order = np.lexsort((-mean_final, np.abs(dev)))
            k = int(order[0])
            _track((np.abs(dev[k]), -mean_final[k]), float(mean_final[k]),
                   float(total[mB[k], mL[k]]), int(mB[k]), int(mL[k]), di)
        else:
            sel = _selection_score(mean_final, dev)
            k = int(np.argmax(sel))
            _track(float(sel[k]), float(mean_final[k]),
                   float(total[mB[k], mL[k]]), int(mB[k]), int(mL[k]), di)

    if best is None:
        return None
    _, mean_final, s, idx = best
    rows = [B.iloc[idx[0]], L.iloc[idx[1]], D.iloc[idx[2]]]
    return mean_final, rows, s


def _meal_entry(row: pd.Series) -> dict:
    return {
        "RecipeId": int(row["RecipeId"]),
        "Name": str(row["Name"]),
        **{col: float(row[col]) for col in NUTRIENT_COLS},
        "final_score": float(row["final_score"]),
    }


def _totals_entry(rows: List[pd.Series]) -> dict:
    return {
        col: round(float(sum(float(r[col]) for r in rows)), 1)
        for col in TOTAL_COLS
    }


def _calorie_cap_message(rules_applied: dict) -> Optional[str]:
    rule = rules_applied.get("Calories")
    if isinstance(rule, tuple) and len(rule) >= 2:
        op, val = rule[0], rule[1]
        if op == "<=":
            return f"calories per recipe at {val} kcal"
        if op == "<":
            return f"calories per recipe below {val} kcal"
        if op == "between" and len(rule) >= 3:
            return f"calories per recipe within {rule[1]}-{rule[2]} kcal"
    return None


def _unreachable_warning(profile: UserProfile, target: float,
                         max_day: float, rules_applied: dict,
                         pool_note: str) -> str:
    cap = _calorie_cap_message(rules_applied)
    gap = (1.0 - max_day / target) * 100 if target > 0 else 0.0
    cond_part = (f" for {', '.join(profile.conditions)}"
                 if profile.conditions else "")
    if cap and target > 0:
        return (
            f"No breakfast+lunch+dinner combination reaches the day target "
            f"({target:.0f} kcal) within the tested tolerance bands. Medical "
            f"rules{cond_part} cap {cap}, so a 3-meal day can reach at most "
            f"~{max_day:.0f} kcal ({gap:.0f}% below the target). Returning the "
            f"closest achievable day instead of forcing an unreachable number."
        )
    return (
        f"No breakfast+lunch+dinner combination reaches the day target "
        f"({target:.0f} kcal) within the tested tolerance bands from the "
        f"available safe ranked pools ({pool_note}). Returning the closest "
        f"achievable day instead of forcing an unreachable number."
    )


def _day_note(tolerance_requested: float, tolerance_used: float,
              reused_slots: List[str]) -> str:
    parts = []
    if tolerance_used is not None and tolerance_used > tolerance_requested + 1e-9:
        parts.append(
            f"no valid combination within +/-{tolerance_requested:.0%} of the "
            f"day target; found within +/-{tolerance_used:.0%}")
    if reused_slots:
        parts.append(
            f"forced reuse of previously assigned recipes for slot(s): "
            f"{', '.join(reused_slots)}")
    return "; ".join(parts)


def _build_slot_pools(system: DietaryExpertSystem, profile: UserProfile,
                      top_k_per_band: int, user_taste_text: Optional[str]):
    _validate_profile(profile)
    pools: Dict[str, pd.DataFrame] = {}
    results: Dict[str, dict] = {}
    ranked_pools: Dict[str, pd.DataFrame] = {}
    target_meal_cal = None
    for slot in MEAL_SLOTS:
        result, ranked = _slot_pipeline(system, profile, slot, user_taste_text)
        results[slot] = result
        ranked_pools[slot] = ranked
        if target_meal_cal is None:
            target_meal_cal = result["target_meal_calories"]
        pools[slot] = _stratified_pool(ranked, top_k_per_band)
    day_target = float(3 * target_meal_cal)
    max_day = float(sum(
        ranked_pools[s]["Calories"].max() if len(ranked_pools[s]) else 0.0
        for s in MEAL_SLOTS))
    return pools, results, day_target, max_day


def _assemble_day(pools: Dict[str, pd.DataFrame], day_target: float,
                  tolerance_requested: float, exclude_ids: set,
                  rules_applied: dict, pool_sizes_full: dict,
                  max_day: float, profile: UserProfile) -> dict:
    band_tolerances = [tolerance_requested]
    widened = min(tolerance_requested + FALLBACK_TOLERANCE_STEP, MAX_TOLERANCE)
    if widened > tolerance_requested + 1e-9:
        band_tolerances.append(widened)

    best = None
    tolerance_used = None
    for tol in band_tolerances:
        searched = {}
        reused_slots = []
        pool_copy = {}
        for slot in MEAL_SLOTS:
            searched_pool, reused = _excluded_pool(pools[slot], exclude_ids)
            pool_copy[slot] = searched_pool
            searched[slot] = len(searched_pool)
            if reused:
                reused_slots.append(slot)
        combo = _combo_search(pool_copy["breakfast"], pool_copy["lunch"],
                              pool_copy["dinner"], day_target, tol)
        if combo is not None:
            best = (combo, tol, reused_slots, searched)
            tolerance_used = tol
            break

    if best is not None:
        (mean_final, rows, s), tol, reused_slots, searched = best
        pool_note = (f"breakfast: {searched['breakfast']}, lunch: "
                     f"{searched['lunch']}, dinner: {searched['dinner']}")
        return {
            "day_target_calories": int(round(day_target)),
            "actual_calories": round(s, 1),
            "target_reached": True,
            "tolerance_requested": tolerance_requested,
            "tolerance_used": tol,
            "warning": "",
            "note": _day_note(tolerance_requested, tol, reused_slots),
            "meals": {slot: _meal_entry(row)
                      for slot, row in zip(MEAL_SLOTS, rows)},
            "totals": _totals_entry(rows),
            "candidate_pools": searched,
            "reused_slots": reused_slots,
        }

    reused_slots = []
    pool_used = {}
    for slot in MEAL_SLOTS:
        pool_used[slot], reused = _excluded_pool(pools[slot], exclude_ids)
        if reused:
            reused_slots.append(slot)
    combo = _combo_search(pool_used["breakfast"], pool_used["lunch"],
                          pool_used["dinner"], day_target, None)
    if combo is None:
        raise RuntimeError(
            "no recipe at all remains for one of the meal slots — the safe "
            "ranked pools are empty (nothing the planner can do; check the "
            "profile's restrictions)")
    _, rows, s = combo
    searched = {slot: len(pool_used[slot]) for slot in MEAL_SLOTS}
    pool_note = (f"breakfast: {searched['breakfast']}, lunch: "
                 f"{searched['lunch']}, dinner: {searched['dinner']}")
    return {
        "day_target_calories": int(round(day_target)),
        "actual_calories": round(s, 1),
        "target_reached": False,
        "tolerance_requested": tolerance_requested,
        "tolerance_used": None,
        "warning": _unreachable_warning(profile, day_target, max_day,
                                        rules_applied, pool_note),
        "note": _day_note(tolerance_requested, tolerance_requested,
                          reused_slots),
        "meals": {slot: _meal_entry(row)
                  for slot, row in zip(MEAL_SLOTS, rows)},
        "totals": _totals_entry(rows),
        "candidate_pools": searched,
        "reused_slots": reused_slots,
    }


def build_daily_plan(profile: UserProfile, system: Optional[DietaryExpertSystem] = None,
                     tolerance: float = DEFAULT_TOLERANCE,
                     top_k_per_band: int = DEFAULT_TOP_K_PER_BAND,
                     exclude_recipe_ids=None,
                     user_taste_text: Optional[str] = None) -> dict:
    _validate_profile(profile)
    _validate_tolerance(tolerance)
    _validate_top_k(top_k_per_band)
    exclude_ids = _validate_exclude(exclude_recipe_ids)
    system = _ensure_system(system)

    pools, results, day_target, max_day = _build_slot_pools(
        system, profile, top_k_per_band, user_taste_text)
    return _assemble_day(pools, day_target, tolerance, exclude_ids,
                         results["breakfast"]["rules_applied"],
                         {slot: len(pools[slot]) for slot in MEAL_SLOTS},
                         max_day, profile)


def build_weekly_plan(profile: UserProfile, days: int = 7,
                      system: Optional[DietaryExpertSystem] = None,
                      tolerance: float = DEFAULT_TOLERANCE,
                      top_k_per_band: int = DEFAULT_TOP_K_PER_BAND,
                      user_taste_text: Optional[str] = None) -> dict:
    _validate_profile(profile)
    _validate_days(days)
    _validate_tolerance(tolerance)
    _validate_top_k(top_k_per_band)
    system = _ensure_system(system)

    pools, results, day_target, max_day = _build_slot_pools(
        system, profile, top_k_per_band, user_taste_text)

    used = set()
    day_results = []
    for day_index in range(1, days + 1):
        day = _assemble_day(pools, day_target, tolerance, used,
                            results["breakfast"]["rules_applied"],
                            {slot: len(pools[slot]) for slot in MEAL_SLOTS},
                            max_day, profile)
        day["day_index"] = day_index
        day_results.append(day)
        used.update(int(m["RecipeId"]) for m in day["meals"].values())

    actuals = [d["actual_calories"] for d in day_results]
    reached = sum(1 for d in day_results if d["target_reached"])
    forced = sum(len(d["reused_slots"]) for d in day_results)
    return {
        "days": day_results,
        "summary": {
            "days_requested": days,
            "days_returned": len(day_results),
            "distinct_recipes_used": len(used),
            "forced_repeat_slots": forced,
            "days_target_reached": reached,
            "avg_target_achievement_pct": round(
                float(np.mean([min(a / day_target, 1.0) * 100 for a in actuals])), 1),
            "avg_actual_calories": round(float(np.mean(actuals)), 1),
        },
    }


if __name__ == "__main__":
    import json as _json
    _profile = UserProfile(
        age=45, height=172.0, weight=88.0, gender="male",
        conditions=["diabetes"], goal="weight_loss", activity_level="light",
    )
    print(_json.dumps(build_daily_plan(_profile), indent=2, ensure_ascii=False))