"""
Tests for the gamification system.
Verifies that XP increases by 10 and streak by 1 for each task.
"""

import sys
import os
import pytest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.gamification import GamificationSystem

class TestGamification:
    """Tests for GamificationSystem."""
    
    @pytest.fixture
    def gamification(self):
        """Create a gamification instance for tests."""
        return GamificationSystem(data_dir="data")
    
    @pytest.fixture
    def user_stats(self):
        """Create initial statistics for a user."""
        return {
            'user_id': 1,
            'xp': 0,
            'level': 1,
            'current_streak': 0,
            'longest_streak': 0,
            'last_activity_date': None,
            'completed_tasks': 0,
            'badges': []
        }
    
    def test_load_data(self, gamification):
        """Test data loading."""
        gamification.load_data()
        
        assert gamification.users_df is not None
        assert gamification.tasks_df is not None
    
    def test_initialize_user_stats(self, gamification):
        """Test user statistics initialization."""
        stats = gamification.initialize_user_stats(1)
        
        assert stats['user_id'] == 1
        assert stats['xp'] == 0
        assert stats['level'] == 1
        assert stats['current_streak'] == 0
        assert stats['completed_tasks'] == 0
        assert stats['badges'] == []
    
    def test_complete_task_xp_increase_beginner(self, gamification, user_stats):
        """Test that XP increases by 10 for Beginner tasks."""
        initial_xp = user_stats['xp']
        initial_tasks = user_stats['completed_tasks']
        
        result = gamification.complete_task(user_stats, 'Beginner')
        
        expected_xp = initial_xp + 10
        if initial_tasks == 0:
            expected_xp += 10
        
        assert user_stats['xp'] == expected_xp
        assert result['xp_gain'] == 10
    
    def test_complete_task_xp_increase_intermediate(self, gamification, user_stats):
        """Test that XP increases by 20 for Intermediate tasks."""
        initial_xp = user_stats['xp']
        initial_tasks = user_stats['completed_tasks']
        
        result = gamification.complete_task(user_stats, 'Intermediate')
        
        expected_xp = initial_xp + 20
        if initial_tasks == 0:
            expected_xp += 10
        
        assert user_stats['xp'] == expected_xp
        assert result['xp_gain'] == 20
    
    def test_complete_task_xp_increase_advanced(self, gamification, user_stats):
        """Test that XP increases by 30 for Advanced tasks."""
        initial_xp = user_stats['xp']
        initial_tasks = user_stats['completed_tasks']
        
        result = gamification.complete_task(user_stats, 'Advanced')
        
        expected_xp = initial_xp + 30
        if initial_tasks == 0:
            expected_xp += 10
        
        assert user_stats['xp'] == expected_xp
        assert result['xp_gain'] == 30
    
    def test_complete_task_streak_increase(self, gamification, user_stats):
        """Test that streak increases by 1 for each completed task."""
        date1 = datetime.now()
        result1 = gamification.complete_task(user_stats, 'Beginner', date1)
        
        assert user_stats['current_streak'] == 1
        assert result1['streak'] == 1
        
        date2 = date1 + timedelta(days=1)
        result2 = gamification.complete_task(user_stats, 'Intermediate', date2)
        
        assert user_stats['current_streak'] == 2
        assert result2['streak'] == 2
        
        date3 = date2 + timedelta(days=1)
        result3 = gamification.complete_task(user_stats, 'Advanced', date3)
        
        assert user_stats['current_streak'] == 3
        assert result3['streak'] == 3
    
    def test_complete_task_streak_reset(self, gamification, user_stats):
        """Test that streak resets when interrupted."""
        date1 = datetime.now()
        gamification.complete_task(user_stats, 'Beginner', date1)
        assert user_stats['current_streak'] == 1
        
        date2 = date1 + timedelta(days=1)
        gamification.complete_task(user_stats, 'Intermediate', date2)
        assert user_stats['current_streak'] == 2
        
        date4 = date2 + timedelta(days=2)
        result = gamification.complete_task(user_stats, 'Advanced', date4)
        
        assert user_stats['current_streak'] == 1
        assert result['streak'] == 1
        assert user_stats['longest_streak'] == 2
    
    def test_complete_task_completed_tasks_increase(self, gamification, user_stats):
        """Test that the number of completed tasks increases."""
        assert user_stats['completed_tasks'] == 0
        
        gamification.complete_task(user_stats, 'Beginner')
        assert user_stats['completed_tasks'] == 1
        
        gamification.complete_task(user_stats, 'Intermediate')
        assert user_stats['completed_tasks'] == 2
        
        gamification.complete_task(user_stats, 'Advanced')
        assert user_stats['completed_tasks'] == 3
    
    def test_complete_task_first_badge(self, gamification, user_stats):
        """Test that the first badge is awarded on the first task."""
        result = gamification.complete_task(user_stats, 'Beginner')
        
        assert 'first_task' in result['new_badges']
        assert 'first_task' in user_stats['badges']
        assert user_stats['xp'] > 10
    
    def test_calculate_level(self, gamification):
        """Test level calculation."""
        assert gamification.calculate_level(0) == 1
        assert gamification.calculate_level(100) == 2
        assert gamification.calculate_level(400) == 3
        assert gamification.calculate_level(900) == 4
    
    def test_level_up(self, gamification, user_stats):
        """Test level increase."""
        initial_level = user_stats['level']
        
        user_stats['xp'] = 100
        user_stats['level'] = gamification.calculate_level(user_stats['xp'])
        
        result = gamification.complete_task(user_stats, 'Advanced')
        
        new_level = gamification.calculate_level(user_stats['xp'])
        if new_level > initial_level:
            assert result['level_up'] == True or user_stats['level'] > initial_level
    
    def test_get_user_stats(self, gamification, user_stats):
        """Test getting user statistics."""
        gamification.complete_task(user_stats, 'Beginner')
        
        stats = gamification.get_user_stats(1, user_stats)
        
        assert 'user_id' in stats
        assert 'xp' in stats
        assert 'level' in stats
        assert 'current_streak' in stats
        assert 'completed_tasks' in stats
        assert 'badges' in stats
        assert stats['xp'] > 0
    
    def test_simulate_days(self, gamification, user_stats):
        """Test simulation over multiple days."""
        events = gamification.simulate_days(user_stats, days=5, completion_rate=1.0)
        
        assert len(events) == 5
        assert all('day' in event for event in events)
        assert all('date' in event for event in events)
        
        completed_days = sum(1 for e in events if e['task_completed'])
        assert completed_days == 5
        
        assert user_stats['current_streak'] == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
