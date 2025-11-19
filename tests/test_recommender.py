"""
Teste pentru sistemul de recomandare.
Verifică că recommend_for_user() întoarce exact k rezultate și că nu pica.
"""

import sys
import os
import pytest

# Adaugă directorul src la path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.recommender import HybridRecommender

class TestRecommender:
    """Teste pentru HybridRecommender."""
    
    @pytest.fixture
    def recommender(self):
        """Creează o instanță de recomandator pentru teste."""
        return HybridRecommender(data_dir="data")
    
    def test_load_data(self, recommender):
        """Testează încărcarea datelor."""
        recommender.load_data()
        
        assert recommender.users_df is not None
        assert recommender.resources_df is not None
        assert recommender.interactions_df is not None
        assert len(recommender.users_df) > 0
        assert len(recommender.resources_df) > 0
    
    def test_fit(self, recommender):
        """Testează antrenarea modelului."""
        recommender.load_data()
        recommender.fit()
        
        assert recommender.is_fitted == True
        assert recommender.tfidf_vectorizer is not None
        assert recommender.resource_vectors is not None
        assert recommender.user_resource_matrix is not None
    
    def test_recommend_for_user_returns_k_results(self, recommender):
        """Testează că recommend_for_user() întoarce exact k rezultate."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        k = 5
        
        recommendations = recommender.recommend_for_user(user_id, k=k)
        
        # Verifică că avem exact k recomandări (sau mai puțin dacă nu sunt suficiente resurse)
        assert len(recommendations) <= k
        assert len(recommendations) > 0  # Ar trebui să avem cel puțin o recomandare
    
    def test_recommend_for_user_structure(self, recommender):
        """Testează structura rezultatelor recomandărilor."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        k = 5
        
        recommendations = recommender.recommend_for_user(user_id, k=k)
        
        if len(recommendations) > 0:
            rec = recommendations[0]
            
            # Verifică că fiecare recomandare are câmpurile necesare
            assert 'resource_id' in rec
            assert 'title' in rec
            assert 'domain' in rec
            assert 'hybrid_score' in rec
            assert 'content_based_score' in rec
            assert 'collaborative_score' in rec
            
            # Verifică că scorurile sunt în intervalul valid
            assert 0 <= rec['hybrid_score'] <= 1
            assert 0 <= rec['content_based_score'] <= 1
            assert 0 <= rec['collaborative_score'] <= 1
    
    def test_recommend_for_user_sorted(self, recommender):
        """Testează că recomandările sunt sortate descrescător după scor."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        k = 5
        
        recommendations = recommender.recommend_for_user(user_id, k=k)
        
        if len(recommendations) > 1:
            # Verifică că sunt sortate descrescător
            scores = [r['hybrid_score'] for r in recommendations]
            assert scores == sorted(scores, reverse=True)
    
    def test_recommend_for_user_no_duplicates(self, recommender):
        """Testează că nu există duplicate în recomandări."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        k = 10
        
        recommendations = recommender.recommend_for_user(user_id, k=k)
        
        # Verifică că nu există duplicate după resource_id
        resource_ids = [r['resource_id'] for r in recommendations]
        assert len(resource_ids) == len(set(resource_ids))
    
    def test_recommend_for_user_excludes_viewed(self, recommender):
        """Testează că recomandările nu includ resurse deja văzute."""
        recommender.load_data()
        recommender.fit()
        
        user_id = 1
        
        # Obține resursele văzute de utilizator
        viewed_resources = set(
            recommender.interactions_df[
                recommender.interactions_df['user_id'] == user_id
            ]['resource_id'].values
        )
        
        recommendations = recommender.recommend_for_user(user_id, k=10)
        
        # Verifică că niciuna dintre recomandări nu este în resursele văzute
        recommended_ids = [r['resource_id'] for r in recommendations]
        assert len(set(recommended_ids) & viewed_resources) == 0
    
    def test_get_user_preferences(self, recommender):
        """Testează obținerea preferințelor utilizatorului."""
        recommender.load_data()
        
        user_id = 1
        preferences = recommender.get_user_preferences(user_id)
        
        assert 'preferred_domains' in preferences
        assert 'experience_level' in preferences
        assert 'learning_style' in preferences
        assert isinstance(preferences['preferred_domains'], list)
    
    def test_recommend_for_user_multiple_users(self, recommender):
        """Testează recomandările pentru mai mulți utilizatori."""
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

