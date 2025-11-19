"""
Sistem de planificare (scheduling) pentru task-uri educaționale.
Implementează EDF (Earliest Deadline First) și CP-SAT cu OR-Tools.
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from ortools.sat.python import cp_model
import os

class Task:
    """Clasă pentru reprezentarea unui task."""
    
    def __init__(self, task_id: int, title: str, deadline: datetime, 
                 estimated_hours: float, priority: str = "Medium"):
        self.task_id = task_id
        self.title = title
        self.deadline = deadline
        self.estimated_hours = estimated_hours
        self.priority = priority
    
    def __repr__(self):
        return f"Task({self.task_id}: {self.title}, deadline={self.deadline.date()}, hours={self.estimated_hours})"

class Scheduler:
    """
    Sistem de planificare care suportă:
    - EDF (Earliest Deadline First) - baseline simplu
    - CP-SAT cu OR-Tools - optimizare avansată
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Inițializează scheduler-ul.
        
        Args:
            data_dir: Directorul unde se află fișierele CSV
        """
        self.data_dir = data_dir
        self.tasks_df = None
    
    def load_tasks(self):
        """Încarcă task-urile din fișierul CSV."""
        tasks_path = os.path.join(self.data_dir, "tasks.csv")
        
        if not os.path.exists(tasks_path):
            raise FileNotFoundError(f"Tasks file not found in {self.data_dir}. Run generate_synthetic_data.py first.")
        
        self.tasks_df = pd.read_csv(tasks_path)
        # Convertește deadline-urile la datetime
        self.tasks_df['deadline'] = pd.to_datetime(self.tasks_df['deadline'])
        print(f"Loaded {len(self.tasks_df)} tasks")
    
    def tasks_to_list(self) -> List[Task]:
        """
        Convertește DataFrame-ul la listă de obiecte Task.
        
        Returns:
            Lista de obiecte Task
        """
        if self.tasks_df is None:
            self.load_tasks()
        
        tasks = []
        for _, row in self.tasks_df.iterrows():
            task = Task(
                task_id=int(row['task_id']),
                title=row['title'],
                deadline=row['deadline'],
                estimated_hours=float(row['estimated_hours']),
                priority=row.get('priority', 'Medium')
            )
            tasks.append(task)
        
        return tasks
    
    def edf_schedule(self, tasks: Optional[List[Task]] = None) -> List[Task]:
        """
        Planificare EDF (Earliest Deadline First) - baseline simplu.
        Sortează task-urile după deadline.
        
        Args:
            tasks: Lista de task-uri (opțional, dacă None, încarcă din CSV)
        
        Returns:
            Lista de task-uri sortată după deadline
        """
        if tasks is None:
            tasks = self.tasks_to_list()
        
        # Sortează după deadline
        sorted_tasks = sorted(tasks, key=lambda t: t.deadline)
        
        return sorted_tasks
    
    def cp_sat_schedule(self, tasks: Optional[List[Task]] = None, 
                       max_hours_per_day: float = 8.0) -> Dict:
        """
        Planificare optimizată cu CP-SAT (OR-Tools).
        Optimizează pentru a minimiza întârzierile și a respecta deadline-urile.
        
        Args:
            tasks: Lista de task-uri (opțional)
            max_hours_per_day: Numărul maxim de ore pe zi
        
        Returns:
            Dicționar cu planificarea optimizată și statistici
        """
        if tasks is None:
            tasks = self.tasks_to_list()
        
        if len(tasks) == 0:
            return {'schedule': [], 'status': 'INFEASIBLE', 'message': 'No tasks to schedule'}
        
        # Creează modelul CP-SAT
        model = cp_model.CpModel()
        
        # Variabile de decizie
        num_tasks = len(tasks)
        start_times = [model.NewIntVar(0, 1000, f'start_{i}') for i in range(num_tasks)]
        end_times = [model.NewIntVar(0, 1000, f'end_{i}') for i in range(num_tasks)]
        
        # Variabile pentru ordinea task-urilor
        # task_order[i][j] = 1 dacă task i este înainte de task j
        task_order = {}
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                task_order[(i, j)] = model.NewBoolVar(f'order_{i}_{j}')
        
        # Constrângeri de bază
        for i in range(num_tasks):
            # end_time = start_time + duration
            model.Add(end_times[i] == start_times[i] + int(tasks[i].estimated_hours))
            
            # Task-ul trebuie să înceapă după momentul curent (ziua 0)
            model.Add(start_times[i] >= 0)
            
            # Deadline constraint (convertim deadline-ul la zile de la momentul curent)
            now = datetime.now()
            deadline_days = (tasks[i].deadline - now).days
            if deadline_days > 0:
                model.Add(end_times[i] <= deadline_days * int(max_hours_per_day))
        
        # Constrângeri de ordine: task-urile nu se suprapun
        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                # Dacă i este înainte de j
                model.Add(start_times[j] >= end_times[i]).OnlyEnforceIf(task_order[(i, j)])
                # Dacă j este înainte de i
                model.Add(start_times[i] >= end_times[j]).OnlyEnforceIf(task_order[(i, j)].Not())
        
        # Funcție obiectiv: minimizăm suma întârzierilor
        # Pentru task-urile care depășesc deadline-ul
        lateness = []
        for i in range(num_tasks):
            now = datetime.now()
            deadline_hours = (tasks[i].deadline - now).total_seconds() / 3600
            if deadline_hours > 0:
                deadline_int = int(deadline_hours)
                lateness_var = model.NewIntVar(0, 1000, f'lateness_{i}')
                model.Add(lateness_var >= end_times[i] - deadline_int)
                lateness.append(lateness_var)
            else:
                lateness.append(model.NewConstant(0))
        
        # Minimizăm suma întârzierilor
        model.Minimize(sum(lateness))
        
        # Rezolvă modelul
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0  # Limită de timp pentru demo
        
        status = solver.Solve(model)
        
        if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
            # Construiește planificarea
            scheduled_tasks = []
            for i in range(num_tasks):
                start_hour = solver.Value(start_times[i])
                end_hour = solver.Value(end_times[i])
                
                # Calculează întârzierea
                now = datetime.now()
                deadline_hours = (tasks[i].deadline - now).total_seconds() / 3600
                lateness_hours = max(0, end_hour - deadline_hours) if deadline_hours > 0 else 0
                
                scheduled_tasks.append({
                    'task': tasks[i],
                    'start_hour': start_hour,
                    'end_hour': end_hour,
                    'lateness_hours': lateness_hours,
                    'on_time': lateness_hours == 0
                })
            
            # Sortează după start_hour
            scheduled_tasks.sort(key=lambda x: x['start_hour'])
            
            return {
                'schedule': scheduled_tasks,
                'status': 'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE',
                'total_lateness': sum(s['lateness_hours'] for s in scheduled_tasks),
                'on_time_tasks': sum(1 for s in scheduled_tasks if s['on_time'])
            }
        else:
            return {
                'schedule': [],
                'status': 'INFEASIBLE',
                'message': 'No feasible solution found'
            }
    
    def compare_schedules(self, edf_tasks: List[Task], cp_sat_result: Dict) -> Dict:
        """
        Compară planificările EDF și CP-SAT.
        
        Args:
            edf_tasks: Lista de task-uri din EDF
            cp_sat_result: Rezultatul din CP-SAT
        
        Returns:
            Dicționar cu comparația
        """
        now = datetime.now()
        
        # Calculează metrici pentru EDF
        edf_lateness = []
        edf_on_time = 0
        for task in edf_tasks:
            deadline_hours = (task.deadline - now).total_seconds() / 3600
            # Simulăm că task-urile sunt făcute secvențial
            cumulative_hours = sum(t.estimated_hours for t in edf_tasks[:edf_tasks.index(task) + 1])
            lateness = max(0, cumulative_hours - deadline_hours) if deadline_hours > 0 else 0
            edf_lateness.append(lateness)
            if lateness == 0:
                edf_on_time += 1
        
        edf_total_lateness = sum(edf_lateness)
        
        # Metrici pentru CP-SAT
        cp_sat_total_lateness = cp_sat_result.get('total_lateness', 0)
        cp_sat_on_time = cp_sat_result.get('on_time_tasks', 0)
        
        return {
            'edf': {
                'total_lateness': edf_total_lateness,
                'on_time_tasks': edf_on_time,
                'total_tasks': len(edf_tasks)
            },
            'cp_sat': {
                'total_lateness': cp_sat_total_lateness,
                'on_time_tasks': cp_sat_on_time,
                'total_tasks': len(cp_sat_result.get('schedule', []))
            },
            'improvement': {
                'lateness_reduction': edf_total_lateness - cp_sat_total_lateness,
                'on_time_improvement': cp_sat_on_time - edf_on_time
            }
        }

