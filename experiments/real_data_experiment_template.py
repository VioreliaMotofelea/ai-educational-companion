"""
Template for running experiments on real educational datasets
(e.g. OULAD, ASSISTments, EdNet).

Datasets are NOT included in the repository due to size and licensing.
The user should download them separately and adjust the paths below.
"""
import pandas as pd
from src.ai_core.recommender import HybridRecommender

def load_oulad(path: str):
    # TODO: implement real preprocessing
    return pd.read_csv(path)

if __name__ == "__main__":
    print("This is a template for future real-data experiments.")
