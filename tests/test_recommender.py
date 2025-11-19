"""
Tests for the recommendation system.
Verifies that recommend_for_user() returns exactly k results and doesn't fail.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.recommender import HybridRecommender

class TestRecommender:
    """Tests for HybridRecommender."""
    
    @pytest.fixture
    def recommender(self):
        """Create a recommender instance for tests."""
        return HybridRecommender(data_dir="data")
    
    def test_load_data(self, recommender):
        """Test data loading."""
        recommender.load_data()
        
        assert recommender.users_df is not None
        assert recommender.resources_df is not None
        assert recommender.interactions_df is not None
        assert len(recommender.users_df) > 0
        assert len(recommender.resources_df) > 0
    
    def test_fit(self, recommender):
        """Test model training."""
        recommender.load_data()
        recommender.fit()
        
        assert recommender.is_fitted == True
        assert recommender.tfidf_vectorizer is not None
        assert recommender.resource_vectors is not None
        assert recommender.user_resource_matrix is not None
    
    def test_recommend_for_user_returns_k_results(self, recommender):
        """Test that recommend_for_user() returns exactly k results."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        k = 5
        
        recommendations = recommender.recommend_for_user(user_id, k=k)
        
        assert len(recommendations) <= k
        assert len(recommendations) > 0
    
    def test_recommend_for_user_structure(self, recommender):
        """Test the structure of recommendation results."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        k = 5
        
        recommendations = recommender.recommend_for_user(user_id, k=k)
        
        if len(recommendations) > 0:
            rec = recommendations[0]
            
            assert 'resource_id' in rec
            assert 'title' in rec
            assert 'domain' in rec
            assert 'hybrid_score' in rec
            assert 'content_based_score' in rec
            assert 'collaborative_score' in rec
            
            assert 0 <= rec['hybrid_score'] <= 1
            assert 0 <= rec['content_based_score'] <= 1
            assert 0 <= rec['collaborative_score'] <= 1
    
    def test_recommend_for_user_sorted(self, recommender):
        """Test that recommendations are sorted in descending order by score."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        k = 5
        
        recommendations = recommender.recommend_for_user(user_id, k=k)
        
        if len(recommendations) > 1:
            scores = [r['hybrid_score'] for r in recommendations]
            assert scores == sorted(scores, reverse=True)
    
    def test_recommend_for_user_no_duplicates(self, recommender):
        """Test that there are no duplicates in recommendations."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        k = 10
        
        recommendations = recommender.recommend_for_user(user_id, k=k)
        
        resource_ids = [r['resource_id'] for r in recommendations]
        assert len(resource_ids) == len(set(resource_ids))
    
    def test_recommend_for_user_excludes_viewed(self, recommender):
        """Test that recommendations don't include already viewed resources."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        
        viewed_resources = set(
            recommender.interactions_df[
                recommender.interactions_df['user_id'] == user_id
            ]['resource_id'].values
        )
        
        recommendations = recommender.recommend_for_user(user_id, k=10)
        
        recommended_ids = [r['resource_id'] for r in recommendations]
        assert len(set(recommended_ids) & viewed_resources) == 0
    
    def test_get_user_preferences(self, recommender):
        """Test getting user preferences."""
        recommender.load_data()
        
        user_id = 1
        preferences = recommender.get_user_preferences(user_id)
        
        assert 'preferred_domains' in preferences
        assert 'experience_level' in preferences
        assert 'learning_style' in preferences
        assert isinstance(preferences['preferred_domains'], list)
    
    def test_recommend_for_user_multiple_users(self, recommender):
        """Test recommendations for multiple users."""
        recommender.load_data()
        recommender.fit()
        
        user_ids = [1, 2, 3]
        k = 5
        
        for user_id in user_ids:
            recommendations = recommender.recommend_for_user(user_id, k=k)
            assert len(recommendations) <= k
            assert len(recommendations) > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
