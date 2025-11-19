"""
AI Core module for the educational companion.
Contains recommender, scheduler, and gamification systems.
"""

from .recommender import HybridRecommender
from .scheduler import Scheduler, Task
from .gamification import GamificationSystem

__all__ = ['HybridRecommender', 'Scheduler', 'Task', 'GamificationSystem']

