"""
Experiment for the gamification system.
Simulates 5 days where the user completes or skips tasks,
and displays XP and streak.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.gamification import GamificationSystem
from datetime import datetime, timedelta

def main():
    """Run the gamification experiment."""
    print("=" * 60)
    print("GAMIFICATION EXPERIMENT")
    print("=" * 60)
    
    print("\n1. Initializing gamification system...")
    gamification = GamificationSystem(data_dir="data")
    
    print("\n2. Loading data...")
    gamification.load_data()
    
    user_id = 1
    print(f"\n3. Simulating 5 days for User {user_id}...")
    print("=" * 60)
    
    user_stats = gamification.initialize_user_stats(user_id)
    
    print(f"\nInitial Stats:")
    print(f"  - XP: {user_stats['xp']}")
    print(f"  - Level: {user_stats['level']}")
    print(f"  - Current Streak: {user_stats['current_streak']}")
    print(f"  - Completed Tasks: {user_stats['completed_tasks']}")
    print(f"  - Badges: {len(user_stats['badges'])}")
    
    print("\n4. Daily Activity Simulation:")
    print("=" * 60)
    
    np.random.seed(42)
    
    events = gamification.simulate_days(user_stats, days=5, completion_rate=0.7)
    
    for event in events:
        print(f"\n--- Day {event['day']} ({event['date']}) ---")
        
        if event['task_completed']:
            print(f"[OK] Task completed!")
            print(f"  Difficulty: {event['difficulty']}")
            print(f"  XP gained: {event['xp_gain']}")
            print(f"  Total XP: {event['new_xp']}")
            print(f"  Level: {event['new_level']}")
            if event['level_up']:
                print(f"  [LEVEL UP!] New level: {event['new_level']}")
            print(f"  Current Streak: {event['streak']} days")
            
            if event['new_badges']:
                print(f"  [BADGES] New Badges Earned:")
                for badge in event['badge_details']:
                    print(f"     - {badge['name']}: {badge['description']}")
        else:
            print("[SKIP] No task completed")
            print(f"  Current Streak: {user_stats['current_streak']} days")
    
    print("\n" + "=" * 60)
    print("FINAL STATISTICS")
    print("=" * 60)
    
    final_stats = gamification.get_user_stats(user_id, user_stats)
    
    print(f"\nUser {user_id} Final Stats:")
    print(f"  - XP: {final_stats['xp']}")
    print(f"  - Level: {final_stats['level']}")
    print(f"  - Current Streak: {final_stats['current_streak']} days")
    print(f"  - Longest Streak: {final_stats['longest_streak']} days")
    print(f"  - Completed Tasks: {final_stats['completed_tasks']}")
    print(f"  - XP to next level: {final_stats['next_level_xp'] - final_stats['xp']}")
    
    print(f"\n  Badges Earned ({len(final_stats['badges'])}):")
    if final_stats['badges']:
        for badge in final_stats['badges']:
            print(f"    - {badge['name']}: {badge['description']}")
    else:
        print("    (No badges yet)")
    
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    
    completed_days = sum(1 for e in events if e['task_completed'])
    total_xp_gained = final_stats['xp']
    badges_earned = len(final_stats['badges'])
    
    print(f"\nDays with completed tasks: {completed_days}/5")
    print(f"Total XP gained: {total_xp_gained}")
    print(f"Level progression: 1 -> {final_stats['level']}")
    print(f"Current streak: {final_stats['current_streak']} days")
    print(f"Badges earned: {badges_earned}")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
