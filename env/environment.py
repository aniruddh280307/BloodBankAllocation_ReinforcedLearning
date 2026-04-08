import random
from typing import List, Tuple, Optional, Dict
import numpy as np
from .models import InventoryState, HospitalRequest, Observation
from .actions import decode_action, validate_allocation_matrix, MAX_REQUESTS, BLOOD_TYPES

MAX_TIMESTEPS = 150
INITIAL_STOCK = {'O': 80, 'A': 60, 'B': 60, 'AB': 40}  # Significantly increased for early-game stability
UNIT_SHELF_DAYS = 7
MAX_REQUESTS_PER_STEP = MAX_REQUESTS
CRITICAL_REQUEST_BUFFER = 5  # Increased from 3 to prevent early termination at step 7
CRITICAL_REQUESTS_THRESHOLD = 8  # Increased from 6 to allow more flexibility

class BloodBankEnv:
    def __init__(self, dataset=None, max_timesteps=MAX_TIMESTEPS, task_id: int = 0):
        self.dataset = dataset
        self.max_timesteps = max_timesteps
        self.task_id = task_id
        self.timestep = 0
        self.inventory = InventoryState()
        self.active_requests: List[HospitalRequest] = []
        self.next_request_id = 0
        self.done = False
        self.score = 0.0
        self.rewards = []
        self.total_fulfilled = 0
        self.total_unfulfilled = 0
        self.total_expired = 0
        self.urgent_fulfilled = 0
        self.total_urgent = 0
        self.critical_steps_count = 0

    def seed(self, s: Optional[int] = None):
        random.seed(s)
        np.random.seed(s)

    def reset(self) -> Observation:
        self.timestep = 0
        self.inventory = InventoryState()
        for t, k in INITIAL_STOCK.items():
            self.inventory.add_units(t, k, UNIT_SHELF_DAYS)
        self.active_requests = []
        for _ in range(3):
            self._add_random_request()
        self.done = False
        self.score = 0.0
        self.rewards = []
        self.total_fulfilled = 0
        self.total_unfulfilled = 0
        self.total_expired = 0
        self.urgent_fulfilled = 0
        self.total_urgent = 0
        self.critical_steps_count = 0
        return self._get_observation()

    def step(self, action: List[int]) -> Tuple[Observation, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Step called after done=True")
        matrix = decode_action(action)
        inv_counts = self.inventory.count_by_type()
        
        # ─── FIX 1: Safety-correct the allocation matrix instead of rejecting ────
        # Clip allocations to valid ranges (don't crash, just fix them)
        matrix = self._safety_correct_allocation_matrix(matrix, inv_counts, self.active_requests)
        
        reward = 0.0
        fulfilled_this_step = 0
        
        for req_idx in range(len(matrix)):
            if req_idx >= len(self.active_requests):
                break
            req = self.active_requests[req_idx]
            alloc_row = matrix[req_idx]
            total_alloc = 0
            for t_idx, alloc in enumerate(alloc_row):
                if alloc <= 0:
                    continue
                donor = BLOOD_TYPES[t_idx]
                removed = self.inventory.remove_units(donor, alloc)
                total_alloc += removed
            if total_alloc >= req.quantity:
                bonus = 2.0 if req.urgency == 2 else 0.0
                reward += 1.0 + bonus
                fulfilled_this_step += 1
                self.total_fulfilled += 1
                if req.urgency == 2:
                    self.urgent_fulfilled += 1
                self.total_urgent += 1
                self.active_requests[req_idx] = None
            else:
                req.quantity -= total_alloc
                if req.urgency == 2:
                    self.total_urgent += 1
        
        self.active_requests = [r for r in self.active_requests if r is not None]
        self.active_requests = [r for r in self.active_requests if r.quantity > 0]

        expired = self.inventory.age_and_expire()
        self.total_expired += expired
        reward -= 0.3 * expired

        unfulfilled_this_step = 0
        for req in self.active_requests:
            # Reduce penalties in VERY early game (steps 1-5) to allow policy ramp-up
            penalty_multiplier = 0.2 if self.timestep <= 5 else 1.0
            
            penalty = 0.2 * penalty_multiplier  # Reduced from 0.5 to be less punitive
            if req.urgency == 2:
                penalty = 0.5 * penalty_multiplier  # Reduced from 1.0
            reward -= penalty
            unfulfilled_this_step += 1
            req.wait_time += 1
            if req.wait_time > 5 and req.urgency < 2:
                req.urgency += 1
        self.total_unfulfilled += unfulfilled_this_step

        reward -= 0.02

        self.timestep += 1

        # ─── FIX 1: Add Inventory Replenishment (simulates blood donations) ────
        # This ensures inventory doesn't completely deplete and the system remains solvable
        self._replenish_inventory()

        # ─── FIX 2: Add new requests (respecting max active request limit) ─────
        # The environment automatically limits to MAX_REQUESTS via _add_random_request
        # Early game introduces requests more gently to allow policy warmup
        if self.timestep <= 2:
            n_new = 0  # Only initial 3 requests for first 2 steps
        elif self.timestep <= 4:
            n_new = 1  # Gentle ramp-up: 1 per step for steps 3-4
        elif self.timestep <= 10:
            n_new = random.randint(1, 1)  # Steps 5-10: 1 per step
        elif self.task_id == 0:
            n_new = random.randint(1, 1)  # Easy: 1 per step
        elif self.task_id == 1:
            n_new = random.randint(1, 2)  # Medium: 1-2 per step
        else:
            n_new = random.randint(1, 2)  # Hard: 1-2 per step (adjusted from 1-3)
        
        # ─── FIX 3: Adaptive demand control - reduce new requests if inventory is critical ─
        inv_counts_current = self.inventory.count_by_type()
        total_inv = sum(inv_counts_current.values())
        pending_qty = sum(req.quantity for req in self.active_requests)
        
        # If inventory is low and many requests pending, skip some new requests
        if total_inv < 20 and pending_qty > 10:
            n_new = max(0, n_new - 1)  # Reduce new requests by 1
        
        for _ in range(n_new):
            self._add_random_request()

        if self.timestep >= self.max_timesteps:
            self.done = True
        
        criticals = sum(1 for r in self.active_requests if r.urgency == 2)
        if criticals >= CRITICAL_REQUESTS_THRESHOLD:
            self.critical_steps_count += 1
        else:
            self.critical_steps_count = 0
        
        if self.critical_steps_count >= CRITICAL_REQUEST_BUFFER:
            # Too many critical requests for too long - mark done but don't penalize
            # The reward is already adjusted by unfulfilled request penalties
            self.done = True

        # ─── FIX 3: Cap extreme reward values to prevent destabilizing spikes ────
        # Keep rewards in a stable range: [-5.0, +10.0]
        # This prevents single-step penalties from destroying the trajectory
        reward = max(-5.0, min(10.0, reward))

        self.rewards.append(reward)
        self.score += reward
        return self._get_observation(), float(reward), self.done, {"fulfilled": fulfilled_this_step, "unfulfilled": unfulfilled_this_step, "expired": expired}

    def compute_score(self) -> float:
        """
        Compute a meaningful performance score based on key metrics.
        
        Returns:
            float: Score in range [0.0, 1.0] reflecting actual performance
        """
        total_requests = max(1, self.total_fulfilled + self.total_unfulfilled)
        
        fulfillment_rate = self.total_fulfilled / total_requests
        urgency_score = self.urgent_fulfilled / max(1, self.total_urgent)
        efficiency_penalty = self.total_expired / max(1, self.total_fulfilled + self.total_expired)
        
        score = (
            0.5 * fulfillment_rate +
            0.3 * urgency_score -
            0.2 * efficiency_penalty
        )
        
        return max(0.0, min(1.0, score))

    def state(self) -> Dict:
        return {
            "timestep": self.timestep,
            "max_timesteps": self.max_timesteps,
            "remaining_timesteps": self.max_timesteps - self.timestep,
            "inventory": self.inventory.count_by_type(),
            "active_requests": [{"type": r.blood_type, "qty": r.quantity, "urg": r.urgency, "wait": r.wait_time} for r in self.active_requests],
            "total_fulfilled": self.total_fulfilled,
            "total_unfulfilled": self.total_unfulfilled,
            "total_expired": self.total_expired,
            "urgent_fulfilled": self.urgent_fulfilled,
            "total_urgent": max(1, self.total_urgent),
            "done": self.done,
            "score": self.compute_score()
        }

    def close(self):
        """Close the environment and return the final computed score."""
        return self.compute_score()

    def _safety_correct_allocation_matrix(self, matrix: List[List[int]], inventory_counts: dict, requests: List) -> List[List[int]]:
        """
        Safely correct allocation matrix without crashing or severe penalties.
        
        Rules:
        1. Clip each allocation to not exceed inventory
        2. Clip each allocation to not exceed request quantity
        3. Remove allocations that violate compatibility rules
        4. Preserve as much valid allocation as possible
        5. FINAL VERIFICATION: zero out entire row if any allocation remains invalid
        
        This is a soft correction - we fix invalid parts instead of rejecting the whole action.
        """
        from .actions import is_compatible, BLOOD_TYPES
        
        corrected = [row[:] for row in matrix]  # Deep copy
        alloc_per_type = {t: 0 for t in BLOOD_TYPES}
        
        for req_idx, row in enumerate(corrected):
            if req_idx >= len(requests):
                # No request for this slot, zero it out
                corrected[req_idx] = [0] * 4
                continue
            
            req = requests[req_idx]
            
            # Check total allocation for this request
            total_alloc = sum(row)
            if total_alloc > req.quantity:
                # Scale down proportionally
                scale = req.quantity / max(1, total_alloc)
                row = [max(0, int(v * scale)) for v in row]
                corrected[req_idx] = row
            
            # Check each blood type allocation
            for t_idx, alloc in enumerate(row):
                donor_type = BLOOD_TYPES[t_idx]
                
                # Check 1: Respect inventory limits
                available = inventory_counts.get(donor_type, 0) - alloc_per_type[donor_type]
                if alloc > available:
                    alloc = max(0, available)
                    corrected[req_idx][t_idx] = alloc
                
                # Check 2: Respect compatibility rules
                if alloc > 0 and not is_compatible(donor_type, req.blood_type):
                    # Zero out incompatible allocation
                    corrected[req_idx][t_idx] = 0
                    alloc = 0
                
                # Track total allocation per type
                alloc_per_type[donor_type] += alloc
        
        # ─── STRICT FINAL VERIFICATION ────
        # After correction, verify ALL allocations are valid
        # If any invalid allocation still exists in a row → zero out entire row
        for req_idx, row in enumerate(corrected):
            if req_idx >= len(requests):
                continue
            
            req = requests[req_idx]
            is_row_valid = True
            
            # Re-check total allocation vs request quantity
            total_alloc = sum(row)
            if total_alloc > req.quantity:
                is_row_valid = False
            
            # Re-check each blood type
            for t_idx, alloc in enumerate(row):
                if alloc < 0:  # Negative allocation
                    is_row_valid = False
                    break
                donor_type = BLOOD_TYPES[t_idx]
                if alloc > 0:
                    # Check compatibility strictly
                    if not is_compatible(donor_type, req.blood_type):
                        is_row_valid = False
                        break
            
            # Re-check inventory totals
            if is_row_valid:
                test_alloc_per_type = {t: 0 for t in BLOOD_TYPES}
                for prev_idx in range(req_idx):
                    for t_idx, v in enumerate(corrected[prev_idx]):
                        test_alloc_per_type[BLOOD_TYPES[t_idx]] += v
                
                for t_idx, alloc in enumerate(row):
                    test_alloc_per_type[BLOOD_TYPES[t_idx]] += alloc
                    if test_alloc_per_type[BLOOD_TYPES[t_idx]] > inventory_counts.get(BLOOD_TYPES[t_idx], 0):
                        is_row_valid = False
                        break
            
            # If any check failed, zero out the entire row
            if not is_row_valid:
                corrected[req_idx] = [0] * 4
        
        return corrected

    def _replenish_inventory(self):
        """
        Add small random blood unit donations each timestep to simulate real-world replenishment.
        
        This prevents inventory depletion and keeps the system solvable throughout the episode.
        Replenishment rates vary by blood type to maintain realistic distributions:
        - O: most common (1-2 units/step)
        - A: moderate (0-1 units/step)
        - B: moderate (0-1 units/step)
        - AB: rare (0-1 units/step)
        """
        # Replenish O-type blood (universal donor, most plentiful)
        units_o = random.randint(1, 2)
        self.inventory.add_units('O', units_o, UNIT_SHELF_DAYS)
        
        # Replenish A-type blood
        units_a = random.randint(0, 1)
        if units_a > 0:
            self.inventory.add_units('A', units_a, UNIT_SHELF_DAYS)
        
        # Replenish B-type blood
        units_b = random.randint(0, 1)
        if units_b > 0:
            self.inventory.add_units('B', units_b, UNIT_SHELF_DAYS)
        
        # Replenish AB-type blood (rarest)
        units_ab = random.randint(0, 1)
        if units_ab > 0:
            self.inventory.add_units('AB', units_ab, UNIT_SHELF_DAYS)

    def _add_random_request(self):
        bt = random.choice(BLOOD_TYPES)
        qty = random.randint(1, 3)
        if self.task_id == 2:
            urg = random.choices([0, 1, 2], [0.2, 0.3, 0.5])[0]
        elif self.task_id == 1:
            urg = random.choices([0, 1, 2], [0.6, 0.3, 0.1])[0]
        else:
            urg = random.choices([0, 1, 2], [0.8, 0.15, 0.05])[0]
        req = HospitalRequest(blood_type=bt, quantity=qty, urgency=urg, wait_time=0, id=self.next_request_id)
        self.next_request_id += 1
        if len(self.active_requests) < MAX_REQUESTS:
            self.active_requests.append(req)

    def _get_observation(self) -> Observation:
        inv_counts = self.inventory.count_by_type()
        counts = [inv_counts[t] for t in BLOOD_TYPES]
        avg_expiry = []
        for t in BLOOD_TYPES:
            vals = [u.days_to_expiry for u in self.inventory.units if u.blood_type == t]
            avg_expiry.append(sum(vals) / len(vals) if vals else 0.0)
        req_features = []
        for i in range(MAX_REQUESTS):
            if i < len(self.active_requests):
                r = self.active_requests[i]
                req_features.append({
                    "blood_type": r.blood_type,
                    "quantity": r.quantity,
                    "urgency": r.urgency,
                    "wait_time": r.wait_time
                })
            else:
                req_features.append({"blood_type": "O", "quantity": 0, "urgency": 0, "wait_time": 0})
        
        return Observation(
            timestep=self.timestep,
            remaining_timesteps=self.max_timesteps - self.timestep,
            inventory_O=counts[0],
            inventory_A=counts[1],
            inventory_B=counts[2],
            inventory_AB=counts[3],
            avg_expiry_O=avg_expiry[0],
            avg_expiry_A=avg_expiry[1],
            avg_expiry_B=avg_expiry[2],
            avg_expiry_AB=avg_expiry[3],
            active_requests=req_features,
            task_id=self.task_id
        )
