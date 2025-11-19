"""
Gamification system for the educational companion.
Implements XP, streaks, and badges.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

class GamificationSystem:
    """
    Gamification system that tracks:
    - XP (Experience Points) - increases when completing tasks
    - Streak - consecutive days of activity
    - Badges - special achievements
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the gamification system.
        
        Args:
            data_dir: Directory where CSV files are located
        """
        self.data_dir = data_dir
        self.users_df = None
        self.tasks_df = None
        
        self.badges = {
            'first_task': {'name': 'First Steps', 'description': 'Complete your first task', 'xp_reward': 10},
            'streak_3': {'name': 'On Fire', 'description': '3 day streak', 'xp_reward': 25},
            'streak_7': {'name': 'Week Warrior', 'description': '7 day streak', 'xp_reward': 50},
            'streak_30': {'name': 'Dedication Master', 'description': '30 day streak', 'xp_reward': 200},
            'xp_100': {'name': 'Centurion', 'description': 'Reach 100 XP', 'xp_reward': 0},
            'xp_500': {'name': 'Half Grand', 'description': 'Reach 500 XP', 'xp_reward': 0},
            'xp_1000': {'name': 'Grand Master', 'description': 'Reach 1000 XP', 'xp_reward': 0},
            'tasks_10': {'name': 'Task Master', 'description': 'Complete 10 tasks', 'xp_reward': 50},
            'tasks_50': {'name': 'Task Legend', 'description': 'Complete 50 tasks', 'xp_reward': 200}
        }
    
    def load_data(self):
        """Load data from CSV files."""
        users_path = os.path.join(self.data_dir, "users.csv")
        tasks_path = os.path.join(self.data_dir, "tasks.csv")
        
        if not os.path.exists(users_path) or not os.path.exists(tasks_path):
            raise FileNotFoundError(f"Data files not found in {self.data_dir}. Run generate_synthetic_data.py first.")
        
        self.users_df = pd.read_csv(users_path)
        self.tasks_df = pd.read_csv(tasks_path)
        if 'created_at' in self.tasks_df.columns:
            self.tasks_df['created_at'] = pd.to_datetime(self.tasks_df['created_at'])
    
    def initialize_user_stats(self, user_id: int) -> Dict:
        """
        Initialize user statistics.
        
        Args:
            user_id: User ID
        
        Returns:
            Dictionary with initial statistics
        """
        return {
            'user_id': user_id,
            'xp': 0,
            'level': 1,
            'current_streak': 0,
            'longest_streak': 0,
            'last_activity_date': None,
            'completed_tasks': 0,
            'badges': []
        }
    
    def calculate_level(self, xp: int) -> int:
        """
        Calculate level based on XP.
        Formula: level = 1 + sqrt(xp / 100)
        
        Args:
            xp: Experience points
        
        Returns:
            User level
        """
        import math
        return 1 + int(math.sqrt(xp / 100))
    
    def complete_task(self, user_stats: Dict, task_difficulty: str = "Medium", 
                     date: Optional[datetime] = None) -> Dict:
        """
        Process task completion and update statistics.
        
        Args:
            user_stats: Current user statistics
            task_difficulty: Task difficulty
            date: Completion date (if None, uses current date)
        
        Returns:
            Dictionary with updates and events (badges earned)
        """
        if date is None:
            date = datetime.now()
        
        xp_rewards = {
            'Beginner': 10,
            'Intermediate': 20,
            'Advanced': 30
        }
        xp_gain = xp_rewards.get(task_difficulty, 15)
        
        old_xp = user_stats['xp']
        user_stats['xp'] += xp_gain
        user_stats['completed_tasks'] += 1
        
        old_level = user_stats['level']
        user_stats['level'] = self.calculate_level(user_stats['xp'])
        level_up = user_stats['level'] > old_level
        
        last_activity = user_stats.get('last_activity_date')
        if last_activity is None:
            user_stats['current_streak'] = 1
        else:
            if isinstance(last_activity, str):
                last_activity = pd.to_datetime(last_activity)
            if isinstance(date, str):
                date = pd.to_datetime(date)
            
            days_diff = (date.date() - last_activity.date()).days
            
            if days_diff == 0:
                pass
            elif days_diff == 1:
                user_stats['current_streak'] += 1
            else:
                if user_stats['current_streak'] > user_stats['longest_streak']:
                    user_stats['longest_streak'] = user_stats['current_streak']
                user_stats['current_streak'] = 1
        
        user_stats['last_activity_date'] = date.isoformat() if isinstance(date, datetime) else date
        
        new_badges = []
        earned_badges = set(user_stats.get('badges', []))
        
        if user_stats['completed_tasks'] == 1 and 'first_task' not in earned_badges:
            new_badges.append('first_task')
            earned_badges.add('first_task')
            user_stats['xp'] += self.badges['first_task']['xp_reward']
        
        streak = user_stats['current_streak']
        if streak >= 3 and 'streak_3' not in earned_badges:
            new_badges.append('streak_3')
            earned_badges.add('streak_3')
            user_stats['xp'] += self.badges['streak_3']['xp_reward']
        if streak >= 7 and 'streak_7' not in earned_badges:
            new_badges.append('streak_7')
            earned_badges.add('streak_7')
            user_stats['xp'] += self.badges['streak_7']['xp_reward']
        if streak >= 30 and 'streak_30' not in earned_badges:
            new_badges.append('streak_30')
            earned_badges.add('streak_30')
            user_stats['xp'] += self.badges['streak_30']['xp_reward']
        
        xp = user_stats['xp']
        if xp >= 100 and 'xp_100' not in earned_badges:
            new_badges.append('xp_100')
            earned_badges.add('xp_100')
        if xp >= 500 and 'xp_500' not in earned_badges:
            new_badges.append('xp_500')
            earned_badges.add('xp_500')
        if xp >= 1000 and 'xp_1000' not in earned_badges:
            new_badges.append('xp_1000')
            earned_badges.add('xp_1000')
        
        completed = user_stats['completed_tasks']
        if completed >= 10 and 'tasks_10' not in earned_badges:
            new_badges.append('tasks_10')
            earned_badges.add('tasks_10')
            user_stats['xp'] += self.badges['tasks_10']['xp_reward']
        if completed >= 50 and 'tasks_50' not in earned_badges:
            new_badges.append('tasks_50')
            earned_badges.add('tasks_50')
            user_stats['xp'] += self.badges['tasks_50']['xp_reward']
        
        user_stats['badges'] = list(earned_badges)
        
        user_stats['level'] = self.calculate_level(user_stats['xp'])
        
        return {
            'xp_gain': xp_gain,
            'new_xp': user_stats['xp'],
            'level_up': level_up,
            'new_level': user_stats['level'],
            'streak': user_stats['current_streak'],
            'new_badges': new_badges,
            'badge_details': [self.badges[badge] for badge in new_badges]
        }
    
    def get_user_stats(self, user_id: int, user_stats: Dict) -> Dict:
        """
        Return complete user statistics.
        
        Args:
            user_id: User ID
            user_stats: User statistics
        
        Returns:
            Dictionary with formatted statistics
        """
        badge_details = [self.badges[badge] for badge in user_stats.get('badges', [])]
        
        return {
            'user_id': user_id,
            'xp': user_stats['xp'],
            'level': user_stats['level'],
            'current_streak': user_stats['current_streak'],
            'longest_streak': user_stats['longest_streak'],
            'completed_tasks': user_stats['completed_tasks'],
            'badges': badge_details,
            'next_level_xp': (user_stats['level'] ** 2) * 100
        }
    
    def simulate_days(self, user_stats: Dict, days: int, 
                     completion_rate: float = 0.7) -> List[Dict]:
        """
        Simulate user activity over multiple days.
        
        Args:
            user_stats: Initial statistics
            days: Number of days to simulate
            completion_rate: Probability of completing a task in a day (0-1)
        
        Returns:
            List with daily events
        """
        events = []
        base_date = datetime.now() - timedelta(days=days)
        
        for day in range(days):
            date = base_date + timedelta(days=day)
            
            if np.random.random() < completion_rate:
                difficulty = np.random.choice(['Beginner', 'Intermediate', 'Advanced'], 
                                                p=[0.4, 0.4, 0.2])
                
                result = self.complete_task(user_stats, difficulty, date)
                events.append({
                    'day': day + 1,
                    'date': date.date().isoformat(),
                    'task_completed': True,
                    'difficulty': difficulty,
                    **result
                })
            else:
                last_activity = user_stats.get('last_activity_date')
                if last_activity:
                    if isinstance(last_activity, str):
                        last_activity = pd.to_datetime(last_activity)
                    days_diff = (date.date() - last_activity.date()).days
                    
                    if days_diff > 1:
                        if user_stats['current_streak'] > user_stats['longest_streak']:
                            user_stats['longest_streak'] = user_stats['current_streak']
                        user_stats['current_streak'] = 0
                
                events.append({
                    'day': day + 1,
                    'date': date.date().isoformat(),
                    'task_completed': False
                })
        
        return events
