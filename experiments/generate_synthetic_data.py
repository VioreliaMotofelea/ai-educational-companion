"""
Script pentru generarea datelor sintetice pentru studiu de caz.
Generează: 10-20 users, 30 tasks, 50 resources, 20-50 interactions.
Salvează în data/users.csv, data/tasks.csv, data/resources.csv, data/interactions.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Seed pentru reproducibilitate
np.random.seed(42)
random.seed(42)

# Domenii educaționale
DOMAINS = ["Computer Science", "Mathematics", "Physics", "Chemistry", "Biology", 
           "Literature", "History", "Art", "Music", "Philosophy"]

# Niveluri de dificultate
DIFFICULTY_LEVELS = ["Beginner", "Intermediate", "Advanced"]

# Tipuri de resurse
RESOURCE_TYPES = ["Video", "Article", "Exercise", "Quiz", "Tutorial", "Book"]

def generate_users(num_users=15):
    """Generează 10-20 utilizatori cu preferințe și profiluri variate."""
    users = []
    
    for i in range(1, num_users + 1):
        # Preferințe: 1-3 domenii preferate
        num_preferences = random.randint(1, 3)
        preferences = random.sample(DOMAINS, num_preferences)
        
        users.append({
            "user_id": i,
            "name": f"User_{i}",
            "email": f"user{i}@example.com",
            "preferred_domains": "|".join(preferences),
            "experience_level": random.choice(["Beginner", "Intermediate", "Advanced"]),
            "learning_style": random.choice(["Visual", "Auditory", "Kinesthetic", "Reading"]),
            "created_at": (datetime.now() - timedelta(days=random.randint(0, 90))).isoformat()
        })
    
    return pd.DataFrame(users)

def generate_tasks(num_tasks=30):
    """Generează 30 de task-uri cu deadline-uri și caracteristici variate."""
    tasks = []
    base_date = datetime.now()
    
    for i in range(1, num_tasks + 1):
        # Deadline între 1-60 zile în viitor
        days_ahead = random.randint(1, 60)
        deadline = base_date + timedelta(days=days_ahead)
        
        # Durata estimată în ore
        estimated_hours = random.uniform(1, 8)
        
        tasks.append({
            "task_id": i,
            "title": f"Task {i}: {random.choice(['Study', 'Complete', 'Review', 'Practice'])} {random.choice(DOMAINS)}",
            "description": f"Complete the following task related to {random.choice(DOMAINS)}",
            "domain": random.choice(DOMAINS),
            "difficulty": random.choice(DIFFICULTY_LEVELS),
            "estimated_hours": round(estimated_hours, 2),
            "deadline": deadline.isoformat(),
            "priority": random.choice(["Low", "Medium", "High"]),
            "status": random.choice(["Pending", "In Progress", "Completed"]),
            "created_at": (base_date - timedelta(days=random.randint(0, 30))).isoformat()
        })
    
    return pd.DataFrame(tasks)

def generate_resources(num_resources=50):
    """Generează 50 de resurse educaționale."""
    resources = []
    
    for i in range(1, num_resources + 1):
        domain = random.choice(DOMAINS)
        resource_type = random.choice(RESOURCE_TYPES)
        
        resources.append({
            "resource_id": i,
            "title": f"{resource_type}: {domain} - Resource {i}",
            "description": f"A comprehensive {resource_type.lower()} about {domain} covering key concepts and practical examples.",
            "domain": domain,
            "resource_type": resource_type,
            "difficulty": random.choice(DIFFICULTY_LEVELS),
            "duration_minutes": random.randint(10, 120),
            "url": f"https://example.com/resources/{i}",
            "tags": "|".join(random.sample(DOMAINS, random.randint(1, 3))),
            "created_at": (datetime.now() - timedelta(days=random.randint(0, 180))).isoformat()
        })
    
    return pd.DataFrame(resources)

def generate_interactions(num_interactions=35, num_users=15, num_resources=50):
    """Generează 20-50 de interacțiuni între utilizatori și resurse."""
    interactions = []
    
    # Asigurăm că fiecare user are cel puțin o interacțiune
    for user_id in range(1, num_users + 1):
        resource_id = random.randint(1, num_resources)
        rating = random.randint(1, 5)
        completion = random.choice([True, False])
        
        interactions.append({
            "interaction_id": len(interactions) + 1,
            "user_id": user_id,
            "resource_id": resource_id,
            "rating": rating,
            "completed": completion,
            "time_spent_minutes": random.randint(5, 60) if completion else random.randint(1, 10),
            "interaction_date": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
        })
    
    # Adăugăm interacțiuni suplimentare până la num_interactions
    while len(interactions) < num_interactions:
        user_id = random.randint(1, num_users)
        resource_id = random.randint(1, num_resources)
        
        # Evităm duplicate-uri
        if not any(i["user_id"] == user_id and i["resource_id"] == resource_id for i in interactions):
            rating = random.randint(1, 5)
            completion = random.choice([True, False])
            
            interactions.append({
                "interaction_id": len(interactions) + 1,
                "user_id": user_id,
                "resource_id": resource_id,
                "rating": rating,
                "completed": completion,
                "time_spent_minutes": random.randint(5, 60) if completion else random.randint(1, 10),
                "interaction_date": (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat()
            })
    
    return pd.DataFrame(interactions)

def main():
    """Funcția principală care generează toate datele și le salvează în CSV."""
    print("Generating synthetic data...")
    
    # Creează directorul data/ dacă nu există
    os.makedirs("data", exist_ok=True)
    
    # Generează datele
    print("  - Generating users...")
    users_df = generate_users(num_users=15)
    
    print("  - Generating tasks...")
    tasks_df = generate_tasks(num_tasks=30)
    
    print("  - Generating resources...")
    resources_df = generate_resources(num_resources=50)
    
    print("  - Generating interactions...")
    interactions_df = generate_interactions(num_interactions=40, num_users=15, num_resources=50)
    
    # Salvează în CSV
    print("\nSaving to CSV files...")
    users_df.to_csv("data/users.csv", index=False)
    print(f"  [OK] Saved {len(users_df)} users to data/users.csv")
    
    tasks_df.to_csv("data/tasks.csv", index=False)
    print(f"  [OK] Saved {len(tasks_df)} tasks to data/tasks.csv")
    
    resources_df.to_csv("data/resources.csv", index=False)
    print(f"  [OK] Saved {len(resources_df)} resources to data/resources.csv")
    
    interactions_df.to_csv("data/interactions.csv", index=False)
    print(f"  [OK] Saved {len(interactions_df)} interactions to data/interactions.csv")
    
    print("\n[SUCCESS] Data generation complete!")
    print(f"\nSummary:")
    print(f"  - Users: {len(users_df)}")
    print(f"  - Tasks: {len(tasks_df)}")
    print(f"  - Resources: {len(resources_df)}")
    print(f"  - Interactions: {len(interactions_df)}")

if __name__ == "__main__":
    main()

