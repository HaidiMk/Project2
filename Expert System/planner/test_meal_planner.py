import sys, json, time
sys.path.insert(0, r'C:\Smart-Dietary-Advisor\Expert System')

from core.user_profile import UserProfile
from engine.filtering_engine import DietaryExpertSystem
from main import load_data
from planner.meal_planner import build_daily_plan, build_weekly_plan

print("تحميل قاعدة البيانات (مرة وحدة، بتاخد كم ثانية)...")
t0 = time.time()
system = DietaryExpertSystem(load_data())
print(f"جاهز خلال {time.time()-t0:.1f} ثانية\n")

profile1 = UserProfile(
    age=45, height=172.0, weight=88.0, gender='male',
    conditions=['diabetes'], goal='weight_loss', activity_level='light'
)

profile2 = UserProfile(
    age=30, height=178.0, weight=75.0, gender='male',
    conditions=['diabetes', 'high_cholesterol'], goal='muscle_gain', activity_level='moderate'
)

print("\n" + "=" * 70)
print("خطة يوم واحد — بروفايل 2 (سكري+كوليسترول، بناء عضل) — متوقع تراجع صادق")
print("=" * 70)
plan2 = build_daily_plan(profile2, system=system)
print(json.dumps(plan2, indent=2, ensure_ascii=False))
print("=" * 70)
print("خطة يوم واحد — بروفايل 1 (سكري، خسارة وزن)")
print("=" * 70)
t0 = time.time()
plan1 = build_daily_plan(profile1, system=system)
print(f"(استغرق {time.time()-t0:.1f} ثانية)\n")
print(json.dumps(plan1, indent=2, ensure_ascii=False))

print("\n" + "=" * 70)
print("خطة أسبوعية كاملة — بروفايل 1 (سكري، خسارة وزن)")
print("=" * 70)
weekly = build_weekly_plan(profile1, days=7, system=system)
print(f"\nالملخص: {weekly['summary']}\n")
for day in weekly['days']:
    names = [m['Name'] for m in day['meals'].values()]
    print(f"  يوم {day['day_index']}: {day['actual_calories']:.0f} سعرة "
          f"| وصل الهدف: {day['target_reached']} | {names}")