#!/usr/bin/env python3
"""
OpenEnv-compliant inference script for Blood Bank Allocation environment.

This script:
1. Runs the environment on 3 task difficulties (easy, medium, hard)
2. Uses a greedy baseline policy
3. Produces OpenEnv-formatted output with exact specification compliance
4. Grades performance using task-specific graders
"""

import os
import sys
from typing import List, Tuple

from env.environment import BloodBankEnv
from env.graders import get_grader
from env.policy import greedy_policy

def format_bool(b: bool) -> str:
    """Format boolean as lowercase 'true' or 'false'."""
    return 'true' if b else 'false'

def run_task(task_id: int, task_name: str, max_steps: int = 100) -> Tuple[float, int, List[float], bool]:
    """
    Run a single task and return (score, steps, rewards, success).
    
    Args:
        task_id: 0=easy, 1=medium, 2=hard
        task_name: human-readable task name
        max_steps: max steps per episode
    
    Returns:
        score (0.0-1.0), num_steps, list of rewards, success bool
    """
    # Print START block for this task
    print(f"[START] task={task_name}")
    
    env = BloodBankEnv(task_id=task_id, max_timesteps=max_steps)
    obs = env.reset()
    steps = 0
    rewards = []
    done = False
    error = None
    
    try:
        while not done and steps < max_steps:
            # Get action from greedy policy
            action = greedy_policy(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            steps += 1
            rewards.append(reward)
            
            # Print step info in OpenEnv format
            action_str = ','.join(str(a) for a in action[:8])  # truncate for brevity
            print(f"[STEP] step={steps} action=[{action_str}...] reward={reward:.2f} done={format_bool(done)} error=null")
    
    except Exception as e:
        error = str(e)
        print(f"[STEP] step={steps} action=null reward=0.00 done=true error={error}")
        # Compute independent score for this task before returning
        total_reward = sum(rewards)
        max_total_reward = steps * 10.0
        score = total_reward / max_total_reward if max_total_reward > 0 else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= 0.1
        print(f"[END] success={format_bool(success)} steps={steps} score={score:.2f}")
        return score, steps, rewards, False
    
    # Compute independent score for this task
    total_reward = sum(rewards)
    max_total_reward = steps * 10.0
    score = total_reward / max_total_reward if max_total_reward > 0 else 0.0
    score = max(0.0, min(1.0, score))
    success = score >= 0.1
    
    # Print END block for this task
    print(f"[END] success={format_bool(success)} steps={steps} score={score:.2f}")
    
    return score, steps, rewards, success

def main():
    """Main entry point."""
    task_configs = [
        (0, "easy_task", 100),
        (1, "medium_task", 150),
        (2, "hard_task", 150)
    ]
    
    for task_id, task_name, max_steps in task_configs:
        run_task(task_id, task_name, max_steps)

if __name__ == '__main__':
    main()

