"""
Tests for the scheduling system.
Verifies that EDF orders tasks in ascending order by deadline.
"""

import sys
import os
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.scheduler import Scheduler, Task

class TestScheduler:
    """Tests for Scheduler."""
    
    @pytest.fixture
    def scheduler(self):
        """Create a scheduler instance for tests."""
        return Scheduler(data_dir="data")
    
    @pytest.fixture
    def sample_tasks(self):
        """Create test tasks."""
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
        """Test loading tasks."""
        scheduler.load_tasks()
        
        assert scheduler.tasks_df is not None
        assert len(scheduler.tasks_df) > 0
        assert 'deadline' in scheduler.tasks_df.columns
    
    def test_tasks_to_list(self, scheduler):
        """Test converting DataFrame to list of Task."""
        scheduler.load_tasks()
        tasks = scheduler.tasks_to_list()
        
        assert len(tasks) > 0
        assert all(isinstance(task, Task) for task in tasks)
        assert all(hasattr(task, 'task_id') for task in tasks)
        assert all(hasattr(task, 'deadline') for task in tasks)
    
    def test_edf_schedule_orders_by_deadline(self, scheduler, sample_tasks):
        """Test that EDF orders tasks in ascending order by deadline."""
        scheduled = scheduler.edf_schedule(sample_tasks)
        
        deadlines = [task.deadline for task in scheduled]
        assert deadlines == sorted(deadlines)
    
    def test_edf_schedule_preserves_all_tasks(self, scheduler, sample_tasks):
        """Test that EDF preserves all tasks."""
        scheduled = scheduler.edf_schedule(sample_tasks)
        
        assert len(scheduled) == len(sample_tasks)
        
        original_ids = {task.task_id for task in sample_tasks}
        scheduled_ids = {task.task_id for task in scheduled}
        assert original_ids == scheduled_ids
    
    def test_edf_schedule_earliest_first(self, scheduler, sample_tasks):
        """Test that the first task in EDF is the one with the nearest deadline."""
        scheduled = scheduler.edf_schedule(sample_tasks)
        
        if len(scheduled) > 0:
            earliest_deadline = min(task.deadline for task in sample_tasks)
            assert scheduled[0].deadline == earliest_deadline
    
    def test_edf_schedule_with_empty_list(self, scheduler):
        """Test EDF with empty list."""
        scheduled = scheduler.edf_schedule([])
        assert len(scheduled) == 0
    
    def test_edf_schedule_with_single_task(self, scheduler):
        """Test EDF with a single task."""
        base_date = datetime.now()
        task = Task(1, "Single Task", base_date + timedelta(days=5), 2.0)
        
        scheduled = scheduler.edf_schedule([task])
        assert len(scheduled) == 1
        assert scheduled[0].task_id == task.task_id
    
    def test_edf_schedule_with_same_deadlines(self, scheduler):
        """Test EDF with tasks that have the same deadline."""
        base_date = datetime.now()
        same_deadline = base_date + timedelta(days=5)
        
        tasks = [
            Task(1, "Task 1", same_deadline, 2.0),
            Task(2, "Task 2", same_deadline, 1.5),
            Task(3, "Task 3", same_deadline, 3.0),
        ]
        
        scheduled = scheduler.edf_schedule(tasks)
        
        assert all(task.deadline == same_deadline for task in scheduled)
        assert len(scheduled) == 3
    
    def test_cp_sat_schedule_structure(self, scheduler, sample_tasks):
        """Test the structure of CP-SAT result."""
        result = scheduler.cp_sat_schedule(sample_tasks, max_hours_per_day=8.0)
        
        assert 'status' in result
        assert 'schedule' in result
        
        if result['status'] in ['OPTIMAL', 'FEASIBLE']:
            assert len(result['schedule']) > 0
            
            for item in result['schedule']:
                assert 'task' in item
                assert 'start_hour' in item
                assert 'end_hour' in item
                assert isinstance(item['task'], Task)
    
    def test_cp_sat_schedule_with_empty_list(self, scheduler):
        """Test CP-SAT with empty list."""
        result = scheduler.cp_sat_schedule([], max_hours_per_day=8.0)
        
        assert result['status'] == 'INFEASIBLE'
        assert len(result['schedule']) == 0
    
    def test_compare_schedules(self, scheduler, sample_tasks):
        """Test schedule comparison."""
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
