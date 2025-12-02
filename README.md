# AI Educational Companion – Research Project

**Author:** Motofelea Viorelia-Maria  
**University:** Babeș-Bolyai University  
**Program:** Computer Science in English  
**Course:** Research Project  

This repository contains the experimental code, datasets, documentation, and testing environment for the **Laboratory deliverable** of the Research Project: An AI-powered educational companion integrating gamification, personalized recommendations, and intelligent scheduling.

The work corresponds to the applied part of the scientific research report and includes:

- Experimental modeling
- Synthetic dataset generation
- Recommender, scheduling, and gamification modules
- Case study experiments
- Validation preparation using real public datasets
- Git-based development history

---

## 📘 Project Overview

This research explores an integrated **AI-powered educational companion** that unifies:

- **Hybrid personalized recommendations** (Content-Based + Collaborative Filtering)
- **Constraint-based intelligent scheduling (CP-SAT with OR-Tools)**
- **Gamification mechanisms** (XP, streaks, badges)
- **Explainability for trust and transparency**

The goal is to improve engagement, productivity, and personalized learning pathways for students.

This repository contains the implementation and experiments used in:

**Lab5_8 – Applied Component of the Research Project.**

---

## 📂 Repository Structure

```
ai-educational-companion/
├── data/                          # Synthetic CSV datasets
│   ├── users.csv                 # User profiles (15 users)
│   ├── tasks.csv                  # Educational tasks (30 tasks)
│   ├── resources.csv              # Learning resources (50 resources)
│   └── interactions.csv           # User-resource interactions (40 interactions)
│
├── src/                           # Source code
│   └── ai_core/                   # Core AI modules
│       ├── __init__.py            # Module exports
│       ├── recommender.py         # Hybrid recommendation system
│       ├── scheduler.py           # Task scheduling (EDF + CP-SAT)
│       └── gamification.py       # Gamification system (XP, streaks, badges)
│
├── experiments/                   # Case study experiments
│   ├── generate_synthetic_data.py    # Data generation script
│   ├── recommender_experiment.py     # Recommender evaluation
│   ├── scheduler_experiment.py        # Scheduling comparison
│   └── gamification_experiment.py    # Gamification simulation
│
├── tests/                         # Unit tests (33 tests)
│   ├── test_recommender.py        # Recommender tests
│   ├── test_scheduler.py          # Scheduler tests
│   └── test_gamification.py       # Gamification tests
│
├── docs/                          # Research documentation
│   ├── Lab3.docx                  # Lab 3 report
│   ├── Lab4.docx                  # Lab 4 report
│   └── Lab5_8_report.docx         # Final lab report
│
├── requirements.txt              # Python dependencies
└── README.md                      # This file
```

---

## 🔧 Core Components

### 1. Hybrid Recommender System (`src/ai_core/recommender.py`)

A hybrid recommendation system combining:

- **Content-Based Filtering (TF-IDF)**
  - Vectorizes resource descriptions using TF-IDF
  - Computes cosine similarity between resources
  - Matches user preferences with resource domains

- **Collaborative Filtering**
  - Builds user-resource rating matrix
  - Finds similar users using cosine similarity
  - Predicts ratings based on user behavior

- **Hybrid Scoring**
  - Combines both approaches with configurable weight (`alpha`)
  - Default: 60% Content-Based, 40% Collaborative
  - Excludes already-viewed resources

**Key Methods:**
- `load_data()` - Loads CSV datasets
- `fit()` - Trains the model (TF-IDF + user matrix)
- `recommend_for_user(user_id, k=5)` - Returns top-k recommendations

### 2. Intelligent Scheduler (`src/ai_core/scheduler.py`)

Task scheduling system with two approaches:

- **EDF (Earliest Deadline First)** - Baseline algorithm
  - Simple sorting by deadline
  - Fast and deterministic
  - Good for basic scenarios

- **CP-SAT Optimization (OR-Tools)** - Advanced algorithm
  - Constraint Programming for optimal scheduling
  - Minimizes lateness
  - Respects deadline constraints and max hours per day
  - Handles task dependencies and overlaps

**Key Methods:**
- `load_tasks()` - Loads tasks from CSV
- `edf_schedule()` - EDF scheduling
- `cp_sat_schedule(max_hours_per_day=8.0)` - CP-SAT optimization
- `compare_schedules()` - Compares EDF vs CP-SAT results

### 3. Gamification System (`src/ai_core/gamification.py`)

Engagement system tracking:

- **Experience Points (XP)**
  - Beginner tasks: +10 XP
  - Intermediate tasks: +20 XP
  - Advanced tasks: +30 XP

- **Streaks**
  - Tracks consecutive days of activity
  - Resets when activity is missed
  - Records longest streak

- **Badges** (9 types)
  - First Steps (first task)
  - On Fire (3-day streak)
  - Week Warrior (7-day streak)
  - Dedication Master (30-day streak)
  - Centurion, Half Grand, Grand Master (XP milestones)
  - Task Master, Task Legend (completion milestones)

- **Levels**
  - Calculated from XP: `level = 1 + sqrt(XP / 100)`
  - Progressive difficulty

**Key Methods:**
- `initialize_user_stats(user_id)` - Creates user profile
- `complete_task(user_stats, difficulty, date)` - Processes task completion
- `simulate_days(user_stats, days, completion_rate)` - Simulates activity
- `get_user_stats(user_id, user_stats)` - Returns formatted statistics

---

## 📊 Synthetic Data Generation

The `generate_synthetic_data.py` script creates a realistic educational dataset:

### Generated Data

- **15 Users**
  - Preferred domains (1-3 per user)
  - Experience levels (Beginner/Intermediate/Advanced)
  - Learning styles (Visual/Auditory/Kinesthetic/Reading)

- **30 Tasks**
  - Various domains (CS, Math, Physics, etc.)
  - Difficulty levels
  - Deadlines (1-60 days ahead)
  - Estimated hours (1-8 hours)
  - Priorities (Low/Medium/High)

- **50 Resources**
  - Types: Video, Article, Exercise, Quiz, Tutorial, Book
  - Domain tags
  - Difficulty levels
  - Duration (10-120 minutes)

- **40 Interactions**
  - User-resource ratings (1-5 stars)
  - Completion status
  - Time spent
  - Interaction dates

### Usage

```bash
python experiments/generate_synthetic_data.py
```

Output: CSV files in `data/` directory

---

## 🧪 Experiments

### 1. Recommender Experiment

**File:** `experiments/recommender_experiment.py`

Evaluates the hybrid recommender system:

- Loads data and trains model
- Generates top-5 recommendations for 2-3 test users
- Computes metrics:
  - Domain match rate (preferred domains)
  - Average hybrid score
  - Content-based vs Collaborative scores

**Metrics:**
- Domain match rate: % of recommendations in user's preferred domains
- Score distribution: Content-based, Collaborative, Hybrid

**Run:**
```bash
python experiments/recommender_experiment.py
```

### 2. Scheduler Experiment

**File:** `experiments/scheduler_experiment.py`

Compares EDF vs CP-SAT scheduling:

- Loads 30 tasks with deadlines
- Runs EDF (baseline) scheduling
- Runs CP-SAT optimization
- Compares results:
  - Total lateness (hours)
  - On-time task count
  - Schedule efficiency

**Run:**
```bash
python experiments/scheduler_experiment.py
```

### 3. Gamification Experiment

**File:** `experiments/gamification_experiment.py`

Simulates 5 days of user activity:

- Initializes user statistics
- Simulates daily task completion (70% completion rate)
- Tracks XP, streaks, badges, levels
- Displays daily events and final statistics

**Run:**
```bash
python experiments/gamification_experiment.py
```

---

## 🧪 Testing

The project includes comprehensive unit tests (33 tests total):

### Test Coverage

- **Recommender Tests** (`tests/test_recommender.py`)
  - Data loading
  - Model fitting
  - Recommendation generation (k results)
  - Score validation
  - Duplicate prevention
  - Excludes viewed resources

- **Scheduler Tests** (`tests/test_scheduler.py`)
  - EDF ordering by deadline
  - Task preservation
  - CP-SAT structure validation
  - Schedule comparison

- **Gamification Tests** (`tests/test_gamification.py`)
  - XP increases (10/20/30 per difficulty)
  - Streak tracking
  - Badge awarding
  - Level calculation

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_recommender.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

**Test Results:** ✅ 33/33 tests passing

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.11+
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ai-educational-companion
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Dependencies:**
   - `numpy>=1.26.0` - Numerical computing
   - `pandas>=2.2.0` - Data manipulation
   - `scikit-learn>=1.4.0` - Machine learning (TF-IDF)
   - `ortools>=9.8.0` - Optimization (CP-SAT)
   - `pytest>=8.1.1` - Testing framework

3. **Generate synthetic data:**
   ```bash
   python experiments/generate_synthetic_data.py
   ```

### Quick Start

```bash
# 1. Generate data
python experiments/generate_synthetic_data.py

# 2. Run experiments
python experiments/recommender_experiment.py
python experiments/scheduler_experiment.py
python experiments/gamification_experiment.py

# 3. Run tests
pytest tests/ -v
```

---

## 📈 Results & Metrics

### Recommender Performance

- **Domain Match Rate:** 40-80% (varies by user)
- **Hybrid Score Range:** 0.50-0.72
- **Content-Based Contribution:** 0.17-0.43
- **Collaborative Contribution:** 0.86-1.00

### Scheduler Performance

- **EDF:** Simple, fast, deterministic
- **CP-SAT:** Optimal scheduling with 0 lateness
- **Comparison:** Both achieve 100% on-time completion for test dataset

### Gamification Engagement

- **XP Progression:** Linear with difficulty-based rewards
- **Streak Tracking:** Encourages daily activity
- **Badge System:** 9 achievement types
- **Level System:** Progressive difficulty scaling

---

## 🔬 Case Study

This implementation serves as a **case study** demonstrating:

1. **Data Collection:** Synthetic dataset generation with realistic educational scenarios
2. **Methodology:** Hybrid approaches combining multiple AI techniques
3. **Validation:** Empirical evidence through experiments and metrics
4. **Interpretation:** Expert analysis of results and system behavior

The case study validates the proposed approach on initial data before scaling to real-world datasets.

---

## 🔮 Future Work

- [ ] Integration with real educational datasets
- [ ] Deep learning enhancements for recommendations
- [ ] Advanced scheduling with resource constraints
- [ ] Explainability module for recommendation transparency
- [ ] User interface for interactive experimentation
- [ ] Performance optimization for larger datasets
- [ ] A/B testing framework for gamification strategies

---

## 📝 Usage Examples

### Using the Recommender

```python
from src.ai_core.recommender import HybridRecommender

recommender = HybridRecommender(data_dir="data")
recommender.load_data()
recommender.fit()

# Get recommendations for user 1
recommendations = recommender.recommend_for_user(user_id=1, k=5)
for rec in recommendations:
    print(f"{rec['title']}: {rec['hybrid_score']:.3f}")
```

### Using the Scheduler

```python
from src.ai_core.scheduler import Scheduler

scheduler = Scheduler(data_dir="data")
scheduler.load_tasks()

# EDF scheduling
edf_schedule = scheduler.edf_schedule()

# CP-SAT optimization
cp_sat_result = scheduler.cp_sat_schedule(max_hours_per_day=8.0)
```

### Using Gamification

```python
from src.ai_core.gamification import GamificationSystem

gamification = GamificationSystem(data_dir="data")
gamification.load_data()

# Initialize user
user_stats = gamification.initialize_user_stats(user_id=1)

# Complete a task
result = gamification.complete_task(user_stats, 'Advanced')
print(f"XP gained: {result['xp_gain']}, New level: {result['new_level']}")
```

---

## 📚 Documentation

- **Lab3.docx** - Initial research and methodology
- **Lab4.docx** - Implementation planning
- **Lab5_8_report.docx** - Final case study report

---

## 📄 License

This project is part of academic research at Babeș-Bolyai University.

---

## 👤 Contact

**Author:** Motofelea Viorelia-Maria  
**Institution:** Babeș-Bolyai University  
**Program:** Computer Science in English

---

## 🙏 Acknowledgments

- Babeș-Bolyai University for research support
- OR-Tools team for optimization framework
- scikit-learn community for ML tools

---

**Last Updated:** November 2025
