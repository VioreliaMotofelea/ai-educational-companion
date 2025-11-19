"""
Hybrid recommendation system for educational resources.
Combines Content-Based Filtering (TF-IDF) with simple Collaborative Filtering.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import os

class HybridRecommender:
    """
    Hybrid recommendation system that combines:
    - Content-Based Filtering (TF-IDF on descriptions)
    - Simple Collaborative Filtering (based on ratings)
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the recommender.
        
        Args:
            data_dir: Directory where CSV files are located
        """
        self.data_dir = data_dir
        self.users_df = None
        self.resources_df = None
        self.interactions_df = None
        self.tfidf_vectorizer = None
        self.resource_vectors = None
        self.user_resource_matrix = None
        self.is_fitted = False
    
    def load_data(self):
        """Load data from CSV files."""
        users_path = os.path.join(self.data_dir, "users.csv")
        resources_path = os.path.join(self.data_dir, "resources.csv")
        interactions_path = os.path.join(self.data_dir, "interactions.csv")
        
        if not all(os.path.exists(p) for p in [users_path, resources_path, interactions_path]):
            raise FileNotFoundError(f"Data files not found in {self.data_dir}. Run generate_synthetic_data.py first.")
        
        self.users_df = pd.read_csv(users_path)
        self.resources_df = pd.read_csv(resources_path)
        self.interactions_df = pd.read_csv(interactions_path)
        
        print(f"Loaded {len(self.users_df)} users, {len(self.resources_df)} resources, {len(self.interactions_df)} interactions")
    
    def _prepare_text_features(self):
        """Prepare text features for TF-IDF vectorization."""
        text_features = []
        for _, row in self.resources_df.iterrows():
            text = f"{row['title']} {row['description']} {row['domain']}"
            if pd.notna(row.get('tags')):
                text += f" {row['tags']}"
            text_features.append(text)
        
        return text_features
    
    def fit(self):
        """
        Train the recommendation model:
        - Vectorize resources with TF-IDF
        - Build user-resource matrix for CF
        """
        if self.users_df is None:
            self.load_data()
        
        print("Vectorizing resources with TF-IDF...")
        text_features = self._prepare_text_features()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.resource_vectors = self.tfidf_vectorizer.fit_transform(text_features)
        
        print("Building user-resource matrix...")
        self.user_resource_matrix = self.interactions_df.pivot_table(
            index='user_id',
            columns='resource_id',
            values='rating',
            fill_value=0
        )
        
        self.is_fitted = True
        print("Model fitted successfully!")
    
    def _content_based_score(self, user_id: int, resource_id: int) -> float:
        """
        Calculate Content-Based score for a resource.
        
        Args:
            user_id: User ID
            resource_id: Resource ID
            
        Returns:
            Content-Based score (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        if user_row.empty:
            return 0.0
        
        preferred_domains = user_row.iloc[0]['preferred_domains'].split('|')
        
        resource_row = self.resources_df[self.resources_df['resource_id'] == resource_id]
        if resource_row.empty:
            return 0.0
        
        resource_domain = resource_row.iloc[0]['domain']
        domain_match = 1.0 if resource_domain in preferred_domains else 0.3
        
        user_interactions = self.interactions_df[
            (self.interactions_df['user_id'] == user_id) & 
            (self.interactions_df['rating'] >= 4)
        ]
        
        if len(user_interactions) == 0:
            return domain_match * 0.5
        
        liked_resources = user_interactions['resource_id'].values
        liked_indices = [idx for idx, rid in enumerate(self.resources_df['resource_id']) if rid in liked_resources]
        
        if len(liked_indices) == 0:
            return domain_match * 0.5
        
        resource_idx = self.resources_df[self.resources_df['resource_id'] == resource_id].index[0]
        resource_vector = self.resource_vectors[resource_idx]
        
        similarities = []
        for liked_idx in liked_indices:
            liked_vector = self.resource_vectors[liked_idx]
            sim = cosine_similarity(resource_vector, liked_vector)[0][0]
            similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return domain_match * (0.5 + 0.5 * avg_similarity)
    
    def _collaborative_filtering_score(self, user_id: int, resource_id: int) -> float:
        """
        Calculate Collaborative Filtering score for a resource.
        
        Args:
            user_id: User ID
            resource_id: Resource ID
            
        Returns:
            CF score (0-1)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_resource_matrix.index:
            return 0.0
        
        if resource_id not in self.user_resource_matrix.columns:
            return 0.0
        
        direct_rating = self.user_resource_matrix.loc[user_id, resource_id]
        if direct_rating > 0:
            return direct_rating / 5.0
        
        user_vector = self.user_resource_matrix.loc[user_id].values
        other_users = self.user_resource_matrix.drop(user_id)
        
        if len(other_users) == 0:
            return 0.0
        
        similarities = []
        ratings = []
        
        for other_user_id, other_user_vector in other_users.iterrows():
            dot_product = np.dot(user_vector, other_user_vector)
            norm_user = np.linalg.norm(user_vector)
            norm_other = np.linalg.norm(other_user_vector)
            
            if norm_user == 0 or norm_other == 0:
                continue
            
            similarity = dot_product / (norm_user * norm_other)
            
            if resource_id in other_user_vector.index:
                rating = other_user_vector[resource_id]
                if rating > 0:
                    similarities.append(similarity)
                    ratings.append(rating)
        
        if len(similarities) == 0 or len(ratings) == 0:
            return 0.0
        
        similarities = np.array(similarities)
        ratings = np.array(ratings)
        
        similarities = (similarities + 1) / 2
        
        weighted_score = np.sum(similarities * ratings) / np.sum(similarities) if np.sum(similarities) > 0 else 0.0
        
        return weighted_score / 5.0
    
    def recommend_for_user(self, user_id: int, k: int = 5, alpha: float = 0.6) -> List[Dict]:
        """
        Generate top-k recommendations for a user.
        
        Args:
            user_id: User ID
            k: Number of recommendations
            alpha: Weight for Content-Based (1-alpha for CF)
        
        Returns:
            List of dictionaries with recommended resources and scores
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        viewed_resources = set(
            self.interactions_df[self.interactions_df['user_id'] == user_id]['resource_id'].values
        )
        
        scores = []
        for _, resource_row in self.resources_df.iterrows():
            resource_id = resource_row['resource_id']
            
            if resource_id in viewed_resources:
                continue
            
            cb_score = self._content_based_score(user_id, resource_id)
            cf_score = self._collaborative_filtering_score(user_id, resource_id)
            
            hybrid_score = alpha * cb_score + (1 - alpha) * cf_score
            
            scores.append({
                'resource_id': resource_id,
                'title': resource_row['title'],
                'domain': resource_row['domain'],
                'resource_type': resource_row['resource_type'],
                'difficulty': resource_row['difficulty'],
                'content_based_score': cb_score,
                'collaborative_score': cf_score,
                'hybrid_score': hybrid_score
            })
        
        scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return scores[:k]
    
    def get_user_preferences(self, user_id: int) -> Dict:
        """Return user preferences."""
        user_row = self.users_df[self.users_df['user_id'] == user_id]
        if user_row.empty:
            return {}
        
        return {
            'preferred_domains': user_row.iloc[0]['preferred_domains'].split('|'),
            'experience_level': user_row.iloc[0]['experience_level'],
            'learning_style': user_row.iloc[0]['learning_style']
        }
