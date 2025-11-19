"""
Experiment for the recommendation system.
Loads data, creates model, generates top-5 recommendations for 2-3 users,
and calculates metrics (e.g., how many recommendations are from preferred domain).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ai_core.recommender import HybridRecommender
import pandas as pd

def main():
    """Run the recommender experiment."""
    print("=" * 60)
    print("RECOMMENDER EXPERIMENT")
    print("=" * 60)
    
    print("\n1. Initializing recommender...")
    recommender = HybridRecommender(data_dir="data")
    
    print("\n2. Loading data...")
    recommender.load_data()
    
    print("\n3. Training model...")
    recommender.fit()
    
    test_users = [1, 2, 3]
    
    print("\n4. Generating recommendations...")
    print("=" * 60)
    
    all_metrics = []
    
    for user_id in test_users:
        print(f"\n--- User {user_id} ---")
        
        preferences = recommender.get_user_preferences(user_id)
        print(f"Preferred domains: {', '.join(preferences['preferred_domains'])}")
        print(f"Experience level: {preferences['experience_level']}")
        print(f"Learning style: {preferences['learning_style']}")
        
        recommendations = recommender.recommend_for_user(user_id, k=5)
        
        print(f"\nTop 5 Recommendations:")
        print("-" * 60)
        
        domain_matches = 0
        for i, rec in enumerate(recommendations, 1):
            is_preferred = rec['domain'] in preferences['preferred_domains']
            if is_preferred:
                domain_matches += 1
            
            print(f"{i}. {rec['title']}")
            print(f"   Domain: {rec['domain']} {'[PREFERRED]' if is_preferred else ''}")
            print(f"   Type: {rec['resource_type']}")
            print(f"   Difficulty: {rec['difficulty']}")
            print(f"   Content-Based Score: {rec['content_based_score']:.3f}")
            print(f"   Collaborative Score: {rec['collaborative_score']:.3f}")
            print(f"   Hybrid Score: {rec['hybrid_score']:.3f}")
            print()
        
        metrics = {
            'user_id': user_id,
            'total_recommendations': len(recommendations),
            'domain_matches': domain_matches,
            'domain_match_rate': domain_matches / len(recommendations) if recommendations else 0,
            'avg_hybrid_score': sum(r['hybrid_score'] for r in recommendations) / len(recommendations) if recommendations else 0,
            'avg_content_score': sum(r['content_based_score'] for r in recommendations) / len(recommendations) if recommendations else 0,
            'avg_collaborative_score': sum(r['collaborative_score'] for r in recommendations) / len(recommendations) if recommendations else 0
        }
        
        all_metrics.append(metrics)
        
        print(f"Metrics:")
        print(f"  - Domain match rate: {metrics['domain_match_rate']:.2%}")
        print(f"  - Average hybrid score: {metrics['avg_hybrid_score']:.3f}")
        print(f"  - Average content-based score: {metrics['avg_content_score']:.3f}")
        print(f"  - Average collaborative score: {metrics['avg_collaborative_score']:.3f}")
    
    print("\n" + "=" * 60)
    print("OVERALL METRICS")
    print("=" * 60)
    
    if all_metrics:
        avg_domain_match = sum(m['domain_match_rate'] for m in all_metrics) / len(all_metrics)
        avg_hybrid_score = sum(m['avg_hybrid_score'] for m in all_metrics) / len(all_metrics)
        
        print(f"\nAverage domain match rate: {avg_domain_match:.2%}")
        print(f"Average hybrid score: {avg_hybrid_score:.3f}")
        
        print("\nDetailed Metrics Table:")
        print("-" * 60)
        metrics_df = pd.DataFrame(all_metrics)
        print(metrics_df.to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Experiment completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
