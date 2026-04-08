from typing import List, Tuple
import numpy as np

# Action encoding: allocation matrix flattened
# For max_requests R and 4 blood types, action is length R*4 integers
# Each value indicates units allocated of that blood type to that request

MAX_REQUESTS = 5
BLOOD_TYPES = ['O', 'A', 'B', 'AB']

class ActionValidationError(Exception):
    pass


def decode_action(action: List[int], max_requests: int = MAX_REQUESTS) -> List[List[int]]:
    """Decode flat action into allocation matrix of shape (num_requests, 4).
    If provided action shorter than max_requests*4, pad with zeros. If longer, truncate.
    """
    action = list(action)
    needed = max_requests * 4
    if len(action) < needed:
        action += [0] * (needed - len(action))
    if len(action) > needed:
        action = action[:needed]
    matrix = []
    for i in range(max_requests):
        row = action[i * 4:(i + 1) * 4]
        matrix.append(row)
    return matrix


def validate_allocation_matrix(matrix: List[List[int]], inventory_counts: dict, requests: List) -> None:
    """Validate allocations:
    - Non-negative integers
    - Do not allocate more than available for each blood type
    - Do not allocate more than requested quantity per request
    - Respect compatibility rules
    """
    # Sum allocated per blood type
    alloc_per_type = {t: 0 for t in BLOOD_TYPES}
    for req_idx, row in enumerate(matrix):
        req = requests[req_idx] if req_idx < len(requests) else None
        for t_idx, v in enumerate(row):
            if not isinstance(v, (int,)):
                raise ActionValidationError("Allocation must be integer")
            if v < 0:
                raise ActionValidationError("Allocation cannot be negative")
            alloc_per_type[BLOOD_TYPES[t_idx]] += v
            # check don't allocate more than requested
            if req is not None:
                total_alloc = sum(row)
                if total_alloc > req.quantity:
                    raise ActionValidationError(f"Allocated more than requested for request {req_idx}")
                # compatibility
                if v > 0:
                    if not is_compatible(BLOOD_TYPES[t_idx], req.blood_type):
                        raise ActionValidationError(f"Blood type {BLOOD_TYPES[t_idx]} not compatible with request {req.blood_type}")
    for t in BLOOD_TYPES:
        if alloc_per_type[t] > inventory_counts.get(t, 0):
            raise ActionValidationError(f"Allocating more units of {t} than available: {alloc_per_type[t]} > {inventory_counts.get(t,0)}")


def is_compatible(donor: str, recipient: str) -> bool:
    """Check if donor blood type can be given to recipient.
    
    ABO Compatibility Rules:
    ┌─────────┬──────────────────────┐
    │ Donor   │ Can give to          │
    ├─────────┼──────────────────────┤
    │ O       │ O, A, B, AB (all)    │ ← universal donor
    │ A       │ A, AB                │
    │ B       │ B, AB                │
    │ AB      │ AB                   │
    └─────────┴──────────────────────┘
    
    Equivalently, recipient needs:
    - O needs: O only
    - A needs: O, A
    - B needs: O, B  
    - AB needs: O, A, B, AB (all) ← universal recipient
    """
    # Define who a specific donor can give to
    can_give_to = {
        'O':  ['O', 'A', 'B', 'AB'],  # O is universal donor
        'A':  ['A', 'AB'],             # A can give to A and AB
        'B':  ['B', 'AB'],             # B can give to B and AB
        'AB': ['AB']                   # AB can only give to AB recipients
    }
    return recipient in can_give_to.get(donor, [])
