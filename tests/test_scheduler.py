"""
Teste pentru sistemul de planificare.
Verifică că EDF ordonează task-urile crescător după deadline.
"""

import sys
import os
import pytest
from datetime import datetime, timedelta

# Adaugă directorul src la path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.scheduler import Scheduler, Task

class TestScheduler:
    """Teste pentru Scheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Creează o instanță de scheduler pentru teste."""
        return Scheduler(data_dir="data")
    
    @pytest.fixture
    def sample_tasks(self):
        """Creează task-uri de test."""
        base_date = datetime.now()
        tasks = [
            Task(1, "Task 1", base_date + timedelta(days=5), 2.0, "High"),
            Task(2, "Task 2", base_date + timedelta(days=3), 1.5, "Medium"),
            Task(3, "Task 3", base_date + timedelta(days=7), 3.0, "Low"),
            Task(4, "Task 4", base_date + timedelta(days=1), 1.0, "High"),
            Task(5, "Task 5", base_date + timedelta(days=10), 4.0, "Medium"),
        ]
        return tasks
    
    def test_load_tasks(self, scheduler):
        """Testează încărcarea task-urilor."""
        scheduler.load_tasks()
        
        assert scheduler.tasks_df is not None
        assert len(scheduler.tasks_df) > 0
        assert 'deadline' in scheduler.tasks_df.columns
    
    def test_tasks_to_list(self, scheduler):
        """Testează conversia DataFrame la listă de Task."""
        scheduler.load_tasks()
        tasks = scheduler.tasks_to_list()
        
        assert len(tasks) > 0
        assert all(isinstance(task, Task) for task in tasks)
        assert all(hasattr(task, 'task_id') for task in tasks)
        assert all(hasattr(task, 'deadline') for task in tasks)
    
    def test_edf_schedule_orders_by_deadline(self, scheduler, sample_tasks):
        """Testează că EDF ordonează task-urile crescător după deadline."""
        scheduled = scheduler.edf_schedule(sample_tasks)
        
        # Verifică că sunt sortate după deadline
        deadlines = [task.deadline for task in scheduled]
        assert deadlines == sorted(deadlines)
    
    def test_edf_schedule_preserves_all_tasks(self, scheduler, sample_tasks):
        """Testează că EDF păstrează toate task-urile."""
        scheduled = scheduler.edf_schedule(sample_tasks)
        
        assert len(scheduled) == len(sample_tasks)
        
        # Verifică că toate task-urile sunt prezente
        original_ids = {task.task_id for task in sample_tasks}
        scheduled_ids = {task.task_id for task in scheduled}
        assert original_ids == scheduled_ids
    
    def test_edf_schedule_earliest_first(self, scheduler, sample_tasks):
        """Testează că primul task din EDF este cel cu deadline-ul cel mai apropiat."""
        scheduled = scheduler.edf_schedule(sample_tasks)
        
        if len(scheduled) > 0:
            earliest_deadline = min(task.deadline for task in sample_tasks)
            assert scheduled[0].deadline == earliest_deadline
    
    def test_edf_schedule_with_empty_list(self, scheduler):
        """Testează EDF cu listă goală."""
        scheduled = scheduler.edf_schedule([])
        assert len(scheduled) == 0
    
    def test_edf_schedule_with_single_task(self, scheduler):
        """Testează EDF cu un singur task."""
        base_date = datetime.now()
        task = Task(1, "Single Task", base_date + timedelta(days=5), 2.0)
        
        scheduled = scheduler.edf_schedule([task])
        assert len(scheduled) == 1
        assert scheduled[0].task_id == task.task_id
    
    def test_edf_schedule_with_same_deadlines(self, scheduler):
        """Testează EDF cu task-uri care au același deadline."""
        base_date = datetime.now()
        same_deadline = base_date + timedelta(days=5)
        
        tasks = [
            Task(1, "Task 1", same_deadline, 2.0),
            Task(2, "Task 2", same_deadline, 1.5),
            Task(3, "Task 3", same_deadline, 3.0),
        ]
        
        scheduled = scheduler.edf_schedule(tasks)
        
        # Toate ar trebui să aibă același deadline
        assert all(task.deadline == same_deadline for task in scheduled)
        assert len(scheduled) == 3
    
    def test_cp_sat_schedule_structure(self, scheduler, sample_tasks):
        """Testează structura rezultatului CP-SAT."""
        result = scheduler.cp_sat_schedule(sample_tasks, max_hours_per_day=8.0)
        
        assert 'status' in result
        assert 'schedule' in result
        
        if result['status'] in ['OPTIMAL', 'FEASIBLE']:
            assert len(result['schedule']) > 0
            
            # Verifică structura fiecărui element din schedule
            for item in result['schedule']:
                assert 'task' in item
                assert 'start_hour' in item
                assert 'end_hour' in item
                assert isinstance(item['task'], Task)
    
    def test_cp_sat_schedule_with_empty_list(self, scheduler):
        """Testează CP-SAT cu listă goală."""
        result = scheduler.cp_sat_schedule([], max_hours_per_day=8.0)
        
        assert result['status'] == 'INFEASIBLE'
        assert len(result['schedule']) == 0
    
    def test_compare_schedules(self, scheduler, sample_tasks):
        """Testează compararea planificărilor."""
        edf_schedule = scheduler.edf_schedule(sample_tasks)
        cp_sat_result = scheduler.cp_sat_schedule(sample_tasks, max_hours_per_day=8.0)
        
        comparison = scheduler.compare_schedules(edf_schedule, cp_sat_result)
        
        assert 'edf' in comparison
        assert 'cp_sat' in comparison
        assert 'improvement' in comparison
        
        assert 'total_lateness' in comparison['edf']
        assert 'on_time_tasks' in comparison['edf']
        assert 'total_tasks' in comparison['edf']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

