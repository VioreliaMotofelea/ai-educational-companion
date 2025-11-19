"""
Teste pentru sistemul de gamificare.
Verifică că XP crește cu 10 și streak cu 1 la fiecare task.
"""

import sys
import os
import pytest
from datetime import datetime, timedelta

# Adaugă directorul src la path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.gamification import GamificationSystem

class TestGamification:
    """Teste pentru GamificationSystem."""
    
    @pytest.fixture
    def gamification(self):
        """Creează o instanță de gamification pentru teste."""
        return GamificationSystem(data_dir="data")
    
    @pytest.fixture
    def user_stats(self):
        """Creează statistici inițiale pentru un utilizator."""
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
        """Testează încărcarea datelor."""
        gamification.load_data()
        
        assert gamification.users_df is not None
        assert gamification.tasks_df is not None
    
    def test_initialize_user_stats(self, gamification):
        """Testează inițializarea statisticilor utilizatorului."""
        stats = gamification.initialize_user_stats(1)
        
        assert stats['user_id'] == 1
        assert stats['xp'] == 0
        assert stats['level'] == 1
        assert stats['current_streak'] == 0
        assert stats['completed_tasks'] == 0
        assert stats['badges'] == []
    
    def test_complete_task_xp_increase_beginner(self, gamification, user_stats):
        """Testează că XP crește cu 10 pentru task-uri Beginner."""
        initial_xp = user_stats['xp']
        initial_tasks = user_stats['completed_tasks']
        
        result = gamification.complete_task(user_stats, 'Beginner')
        
        # XP-ul task-ului este 10, dar primul task primește și badge "first_task" (+10 XP)
        expected_xp = initial_xp + 10
        if initial_tasks == 0:
            expected_xp += 10  # Badge bonus pentru primul task
        
        assert user_stats['xp'] == expected_xp
        assert result['xp_gain'] == 10  # XP-ul task-ului în sine
    
    def test_complete_task_xp_increase_intermediate(self, gamification, user_stats):
        """Testează că XP crește cu 20 pentru task-uri Intermediate."""
        initial_xp = user_stats['xp']
        initial_tasks = user_stats['completed_tasks']
        
        result = gamification.complete_task(user_stats, 'Intermediate')
        
        # XP-ul task-ului este 20, dar primul task primește și badge "first_task" (+10 XP)
        expected_xp = initial_xp + 20
        if initial_tasks == 0:
            expected_xp += 10  # Badge bonus pentru primul task
        
        assert user_stats['xp'] == expected_xp
        assert result['xp_gain'] == 20  # XP-ul task-ului în sine
    
    def test_complete_task_xp_increase_advanced(self, gamification, user_stats):
        """Testează că XP crește cu 30 pentru task-uri Advanced."""
        initial_xp = user_stats['xp']
        initial_tasks = user_stats['completed_tasks']
        
        result = gamification.complete_task(user_stats, 'Advanced')
        
        # XP-ul task-ului este 30, dar primul task primește și badge "first_task" (+10 XP)
        expected_xp = initial_xp + 30
        if initial_tasks == 0:
            expected_xp += 10  # Badge bonus pentru primul task
        
        assert user_stats['xp'] == expected_xp
        assert result['xp_gain'] == 30  # XP-ul task-ului în sine
    
    def test_complete_task_streak_increase(self, gamification, user_stats):
        """Testează că streak-ul crește cu 1 la fiecare task completat."""
        # Prima activitate
        date1 = datetime.now()
        result1 = gamification.complete_task(user_stats, 'Beginner', date1)
        
        assert user_stats['current_streak'] == 1
        assert result1['streak'] == 1
        
        # A doua activitate în ziua următoare
        date2 = date1 + timedelta(days=1)
        result2 = gamification.complete_task(user_stats, 'Intermediate', date2)
        
        assert user_stats['current_streak'] == 2
        assert result2['streak'] == 2
        
        # A treia activitate în ziua următoare
        date3 = date2 + timedelta(days=1)
        result3 = gamification.complete_task(user_stats, 'Advanced', date3)
        
        assert user_stats['current_streak'] == 3
        assert result3['streak'] == 3
    
    def test_complete_task_streak_reset(self, gamification, user_stats):
        """Testează că streak-ul se resetează când se întrerupe."""
        # Prima activitate
        date1 = datetime.now()
        gamification.complete_task(user_stats, 'Beginner', date1)
        assert user_stats['current_streak'] == 1
        
        # A doua activitate în ziua următoare
        date2 = date1 + timedelta(days=1)
        gamification.complete_task(user_stats, 'Intermediate', date2)
        assert user_stats['current_streak'] == 2
        
        # Sare peste o zi (întrerupe streak-ul)
        date4 = date2 + timedelta(days=2)
        result = gamification.complete_task(user_stats, 'Advanced', date4)
        
        assert user_stats['current_streak'] == 1
        assert result['streak'] == 1
        assert user_stats['longest_streak'] == 2
    
    def test_complete_task_completed_tasks_increase(self, gamification, user_stats):
        """Testează că numărul de task-uri completate crește."""
        assert user_stats['completed_tasks'] == 0
        
        gamification.complete_task(user_stats, 'Beginner')
        assert user_stats['completed_tasks'] == 1
        
        gamification.complete_task(user_stats, 'Intermediate')
        assert user_stats['completed_tasks'] == 2
        
        gamification.complete_task(user_stats, 'Advanced')
        assert user_stats['completed_tasks'] == 3
    
    def test_complete_task_first_badge(self, gamification, user_stats):
        """Testează că primul badge este acordat la primul task."""
        result = gamification.complete_task(user_stats, 'Beginner')
        
        assert 'first_task' in result['new_badges']
        assert 'first_task' in user_stats['badges']
        # Verifică că badge-ul dă XP bonus
        assert user_stats['xp'] > 10  # Mai mult decât doar XP-ul task-ului
    
    def test_calculate_level(self, gamification):
        """Testează calculul nivelului."""
        assert gamification.calculate_level(0) == 1
        assert gamification.calculate_level(100) == 2
        assert gamification.calculate_level(400) == 3
        assert gamification.calculate_level(900) == 4
    
    def test_level_up(self, gamification, user_stats):
        """Testează creșterea nivelului."""
        # Completează task-uri până când nivelul crește
        initial_level = user_stats['level']
        
        # Adaugă XP suficient pentru level up
        user_stats['xp'] = 100
        user_stats['level'] = gamification.calculate_level(user_stats['xp'])
        
        result = gamification.complete_task(user_stats, 'Advanced')
        
        new_level = gamification.calculate_level(user_stats['xp'])
        if new_level > initial_level:
            assert result['level_up'] == True or user_stats['level'] > initial_level
    
    def test_get_user_stats(self, gamification, user_stats):
        """Testează obținerea statisticilor utilizatorului."""
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
        """Testează simularea pe mai multe zile."""
        events = gamification.simulate_days(user_stats, days=5, completion_rate=1.0)
        
        assert len(events) == 5
        assert all('day' in event for event in events)
        assert all('date' in event for event in events)
        
        # Cu completion_rate=1.0, toate zilele ar trebui să aibă task-uri completate
        completed_days = sum(1 for e in events if e['task_completed'])
        assert completed_days == 5
        
        # Verifică că streak-ul crește
        assert user_stats['current_streak'] == 5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

