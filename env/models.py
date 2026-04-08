from dataclasses import dataclass, field
from typing import List, Dict
from pydantic import BaseModel

# Pydantic OpenEnv Models (required for OpenEnv compliance)

class Observation(BaseModel):
    """Observation from the blood bank environment."""
    timestep: int
    remaining_timesteps: int
    inventory_O: int
    inventory_A: int
    inventory_B: int
    inventory_AB: int
    avg_expiry_O: float
    avg_expiry_A: float
    avg_expiry_B: float
    avg_expiry_AB: float
    active_requests: List[Dict]  # list of {blood_type, quantity, urgency, wait_time}
    task_id: int = 0

class Action(BaseModel):
    """Action: allocation matrix [request_idx][blood_type_idx] -> units to allocate."""
    allocations: List[List[int]]  # 5x4 matrix (max 5 requests, 4 blood types)

class Reward(BaseModel):
    """Reward breakdown."""
    value: float
    fulfilled_requests: int = 0
    unfulfilled_requests: int = 0
    expired_units: int = 0

# Legacy dataclasses for internal use
@dataclass
class BloodUnit:
    blood_type: str
    days_to_expiry: int

@dataclass
class HospitalRequest:
    blood_type: str
    quantity: int
    urgency: int
    wait_time: int = 0
    id: int = -1

@dataclass
class InventoryState:
    units: List[BloodUnit] = field(default_factory=list)

    def count_by_type(self):
        counts = {t: 0 for t in ['O', 'A', 'B', 'AB']}
        for u in self.units:
            counts[u.blood_type] += 1
        return counts

    def remove_units(self, blood_type: str, k: int) -> int:
        removed = 0
        remaining = []
        for u in sorted(self.units, key=lambda x: x.days_to_expiry):
            if removed < k and u.blood_type == blood_type:
                removed += 1
                continue
            remaining.append(u)
        self.units = remaining
        return removed

    def age_and_expire(self) -> int:
        expired = 0
        for u in self.units:
            u.days_to_expiry -= 1
        still = []
        for u in self.units:
            if u.days_to_expiry <= 0:
                expired += 1
            else:
                still.append(u)
        self.units = still
        return expired

    def add_units(self, blood_type: str, k: int, days_to_expiry: int):
        for _ in range(k):
            self.units.append(BloodUnit(blood_type=blood_type, days_to_expiry=days_to_expiry))

    def total_units(self):
        return len(self.units)
