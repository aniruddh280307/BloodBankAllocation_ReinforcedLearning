"""Task graders for the 3 difficulty levels."""

class TaskGrader:
    """Base grader interface."""
    def grade(self, state: dict) -> float:
        """Return score 0.0-1.0."""
        raise NotImplementedError

class EasyTaskGrader(TaskGrader):
    """EASY TASK: Minimal wastage with moderate demand.
    
    Objective: Manage a blood bank with low request volume and low urgency.
    Success: Fulfill at least 60% of requests with <2 expired units per episode.
    Grading:
    - Base: (fulfilled / (fulfilled + unfulfilled)) * 0.5
    - Wastage penalty: -0.1 per expired unit (capped at 0.2)
    - Time efficiency: remaining_time / max_time * 0.3 (bonus for finishing early)
    Score: max(0, base + bonus - penalty)
    """
    def grade(self, state: dict) -> float:
        fulfilled = state.get('total_fulfilled', 0)
        unfulfilled = state.get('total_unfulfilled', 0)
        expired = state.get('total_expired', 0)
        remaining = state.get('remaining_timesteps', 0)
        max_ts = state.get('max_timesteps', 150)
        
        if fulfilled + unfulfilled == 0:
            return 0.0
        
        fulfillment_rate = fulfilled / (fulfilled + unfulfilled)
        base = fulfillment_rate * 0.5
        
        wastage_penalty = min(0.2, expired * 0.1)
        time_bonus = (remaining / max_ts) * 0.3
        
        score = base + time_bonus - wastage_penalty
        return max(0.0, min(1.0, score))

class MediumTaskGrader(TaskGrader):
    """MEDIUM TASK: Balanced allocation with increasing demand and mixed urgency.
    
    Objective: Handle moderate request volume with varied urgency levels.
    Success: Fulfill >70% of requests, prioritize urgent cases (urgency=2).
    Grading:
    - Fulfillment: (fulfilled / (fulfilled + unfulfilled)) * 0.4
    - Urgency handling: (urgent_fulfilled / total_urgent) * 0.3 (if any urgent)
    - Wastage: -0.05 per expired unit (capped at 0.15)
    - Consistency: reward long streaks without critical failures
    Score: max(0, base + urgency + bonus - penalty)
    """
    def grade(self, state: dict) -> float:
        fulfilled = state.get('total_fulfilled', 0)
        unfulfilled = state.get('total_unfulfilled', 0)
        expired = state.get('total_expired', 0)
        urgent_fulfilled = state.get('urgent_fulfilled', 0)
        total_urgent = state.get('total_urgent', 1)
        done = state.get('done', False)
        
        if fulfilled + unfulfilled == 0:
            return 0.0
        
        fulfillment_rate = fulfilled / (fulfilled + unfulfilled)
        base = fulfillment_rate * 0.4
        
        urgency_rate = (urgent_fulfilled / max(total_urgent, 1))
        urgency_bonus = urgency_rate * 0.3
        
        wastage_penalty = min(0.15, expired * 0.05)
        system_failure_penalty = 0.2 if done and unfulfilled > 5 else 0.0
        
        score = base + urgency_bonus - wastage_penalty - system_failure_penalty
        return max(0.0, min(1.0, score))

class HardTaskGrader(TaskGrader):
    """HARD TASK: Complex allocation with high demand and frequent urgent requests.
    
    Objective: Manage high-pressure scenarios with severe time pressure.
    Success: Maintain <10% critical request backlog; fulfill urgent cases with priority.
    Grading:
    - Fulfillment: (fulfilled / (fulfilled + unfulfilled)) * 0.35
    - Urgency performance: (urgent_met / total_urgent) * 0.35 (critical!)
    - Efficiency: -0.02 per expired unit + -0.01 per pending request (capped at 0.3)
    - Crisis management: +0.1 if no system failure (done=True without crash)
    Score: max(0, base + urgency + efficiency + crisis_bonus)
    """
    def grade(self, state: dict) -> float:
        fulfilled = state.get('total_fulfilled', 0)
        unfulfilled = state.get('total_unfulfilled', 0)
        expired = state.get('total_expired', 0)
        urgent_fulfilled = state.get('urgent_fulfilled', 0)
        total_urgent = state.get('total_urgent', 1)
        done = state.get('done', False)
        score = state.get('score', 0)
        
        if fulfilled + unfulfilled == 0:
            return 0.0
        
        fulfillment_rate = fulfilled / (fulfilled + unfulfilled)
        base = fulfillment_rate * 0.35
        
        urgency_rate = (urgent_fulfilled / max(total_urgent, 1))
        urgency_bonus = urgency_rate * 0.35
        
        efficiency_penalty = min(0.3, (expired * 0.02) + (unfulfilled * 0.01))
        crisis_bonus = 0.1 if done and score > 0 else 0.0
        
        score_final = base + urgency_bonus + crisis_bonus - efficiency_penalty
        return max(0.0, min(1.0, score_final))

def get_grader(task_id: int) -> TaskGrader:
    """Get grader for task difficulty."""
    if task_id == 0:
        return EasyTaskGrader()
    elif task_id == 1:
        return MediumTaskGrader()
    else:
        return HardTaskGrader()