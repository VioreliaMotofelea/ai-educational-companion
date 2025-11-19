"""
Experiment for the scheduling system.
Loads tasks from CSV, runs EDF, displays order,
and compares with CP-SAT scheduling.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.scheduler import Scheduler
from datetime import datetime

def main():
    """Run the scheduler experiment."""
    print("=" * 60)
    print("SCHEDULER EXPERIMENT")
    print("=" * 60)
    
    print("\n1. Initializing scheduler...")
    scheduler = Scheduler(data_dir="data")
    
    print("\n2. Loading tasks...")
    scheduler.load_tasks()
    
    tasks = scheduler.tasks_to_list()
    print(f"Loaded {len(tasks)} tasks")
    
    print("\n3. Running EDF (Earliest Deadline First) scheduling...")
    print("=" * 60)
    edf_schedule = scheduler.edf_schedule(tasks)
    
    print("\nEDF Schedule (ordered by deadline):")
    print("-" * 60)
    now = datetime.now()
    cumulative_hours = 0
    
    for i, task in enumerate(edf_schedule, 1):
        cumulative_hours += task.estimated_hours
        deadline_days = (task.deadline - now).days
        deadline_hours = deadline_days * 8 if deadline_days > 0 else 0
        
        on_time = "[OK]" if cumulative_hours <= deadline_hours else "[LATE]"
        lateness = max(0, cumulative_hours - deadline_hours) if deadline_hours > 0 else 0
        
        print(f"{i:2d}. Task {task.task_id}: {task.title[:40]}")
        print(f"     Deadline: {task.deadline.strftime('%Y-%m-%d')} ({deadline_days} days)")
        print(f"     Estimated hours: {task.estimated_hours:.2f}")
        print(f"     Cumulative hours: {cumulative_hours:.2f}")
        print(f"     Status: {on_time} {'On time' if on_time == '[OK]' else f'Late by {lateness:.2f}h'}")
        print()
    
    print("\n4. Running CP-SAT optimization...")
    print("=" * 60)
    cp_sat_result = scheduler.cp_sat_schedule(tasks, max_hours_per_day=8.0)
    
    if cp_sat_result['status'] in ['OPTIMAL', 'FEASIBLE']:
        print(f"\nCP-SAT Schedule ({cp_sat_result['status']}):")
        print("-" * 60)
        
        for i, item in enumerate(cp_sat_result['schedule'], 1):
            task = item['task']
            start_hour = item['start_hour']
            end_hour = item['end_hour']
            lateness = item['lateness_hours']
            on_time = "[OK]" if item['on_time'] else "[LATE]"
            
            start_day = int(start_hour // 8) + 1
            start_hour_in_day = start_hour % 8
            
            print(f"{i:2d}. Task {task.task_id}: {task.title[:40]}")
            print(f"     Start: Day {start_day}, Hour {start_hour_in_day:.2f}")
            print(f"     End: Hour {end_hour:.2f}")
            print(f"     Deadline: {task.deadline.strftime('%Y-%m-%d')}")
            print(f"     Status: {on_time} {'On time' if on_time == '[OK]' else f'Late by {lateness:.2f}h'}")
            print()
        
        print(f"Total lateness: {cp_sat_result['total_lateness']:.2f} hours")
        print(f"On-time tasks: {cp_sat_result['on_time_tasks']}/{len(cp_sat_result['schedule'])}")
    else:
        print(f"\nCP-SAT Status: {cp_sat_result['status']}")
        print(f"Message: {cp_sat_result.get('message', 'N/A')}")
    
    print("\n5. Comparing EDF vs CP-SAT...")
    print("=" * 60)
    
    comparison = scheduler.compare_schedules(edf_schedule, cp_sat_result)
    
    print("\nComparison Results:")
    print("-" * 60)
    print("EDF:")
    print(f"  - Total lateness: {comparison['edf']['total_lateness']:.2f} hours")
    print(f"  - On-time tasks: {comparison['edf']['on_time_tasks']}/{comparison['edf']['total_tasks']}")
    
    if cp_sat_result['status'] in ['OPTIMAL', 'FEASIBLE']:
        print("\nCP-SAT:")
        print(f"  - Total lateness: {comparison['cp_sat']['total_lateness']:.2f} hours")
        print(f"  - On-time tasks: {comparison['cp_sat']['on_time_tasks']}/{comparison['cp_sat']['total_tasks']}")
        
        print("\nImprovement:")
        lateness_reduction = comparison['improvement']['lateness_reduction']
        on_time_improvement = comparison['improvement']['on_time_improvement']
        
        if lateness_reduction > 0:
            print(f"  - Lateness reduced by: {lateness_reduction:.2f} hours ({lateness_reduction/comparison['edf']['total_lateness']*100:.1f}%)")
        else:
            print(f"  - Lateness increased by: {abs(lateness_reduction):.2f} hours")
        
        if on_time_improvement > 0:
            print(f"  - On-time tasks increased by: {on_time_improvement}")
        elif on_time_improvement < 0:
            print(f"  - On-time tasks decreased by: {abs(on_time_improvement)}")
        else:
            print(f"  - No change in on-time tasks")
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
