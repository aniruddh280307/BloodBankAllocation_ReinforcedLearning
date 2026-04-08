"""Baseline policy: improved greedy allocation strategy."""
from typing import List


def greedy_policy(obs) -> List[int]:
    """
    Improved greedy allocation strategy with balanced distribution:
    
    1. Sort requests by priority: urgency DESC → wait_time DESC → quantity DESC
    2. First pass: limited balanced allocation (max 2 units per request per blood type)
    3. Second pass: fulfill remaining high-priority requests fully
    4. Zero-action guard: ensure at least 1 unit allocated if inventory exists
    5. O-type reserve for future critical requests
    
    Returns: flattened allocation matrix (5x4 = 20 values)
    """
    # Parse observation
    requests   = obs.active_requests if hasattr(obs, 'active_requests') else obs.get('active_requests', [])
    inventory_O  = obs.inventory_O  if hasattr(obs, 'inventory_O')  else obs.get('inventory_O',  0)
    inventory_A  = obs.inventory_A  if hasattr(obs, 'inventory_A')  else obs.get('inventory_A',  0)
    inventory_B  = obs.inventory_B  if hasattr(obs, 'inventory_B')  else obs.get('inventory_B',  0)
    inventory_AB = obs.inventory_AB if hasattr(obs, 'inventory_AB') else obs.get('inventory_AB', 0)

    inventories = {'O': inventory_O, 'A': inventory_A, 'B': inventory_B, 'AB': inventory_AB}
    blood_types  = ['O', 'A', 'B', 'AB']
    max_requests = 5

    # Dead-zone guard: no inventory = no allocation possible
    if sum(inventories.values()) == 0:
        return [0] * (max_requests * 4)

    # Build list of active requests (quantity > 0)
    active_reqs = []
    for i, r in enumerate(requests):
        if isinstance(r, dict):
            qty        = r.get('quantity',   0)
            urgency    = r.get('urgency',    0)
            wait       = r.get('wait_time',  0)
            blood_type = r.get('blood_type', 'O')
        else:
            qty        = getattr(r, 'quantity',   0)
            urgency    = getattr(r, 'urgency',    0)
            wait       = getattr(r, 'wait_time',  0)
            blood_type = getattr(r, 'blood_type', 'O')

        if qty > 0:
            active_reqs.append((i, qty, urgency, wait, blood_type))

    # Dead-zone guard: no pending requests = no allocation needed
    if not active_reqs:
        return [0] * (max_requests * 4)

    # Sort by priority: urgency DESC → wait_time DESC → quantity DESC
    active_reqs.sort(key=lambda x: (-x[2], -x[3], -x[1]))

    allocation_matrix = [[0] * 4 for _ in range(max_requests)]
    inventories_copy = inventories.copy()

    # Reserve O-type buffer for future critical requests
    o_reserve = max(3, inventory_O // 5)
    available_o = max(0, inventories_copy['O'] - o_reserve)

    # ─── FIRST PASS: Balanced allocation (max 2 units per request per blood type) ────
    for orig_idx, quantity, urgency, wait_time, blood_type in active_reqs:
        if orig_idx >= max_requests or quantity <= 0:
            continue

        compatible = get_compatible_donors(blood_type)
        remaining = quantity

        for donor in compatible:
            if remaining <= 0:
                break
            
            donor_idx = blood_types.index(donor)
            
            # Limit per-request allocation in first pass
            if donor == 'O':
                available = available_o
            else:
                available = inventories_copy[donor]
            
            # Key change: limit to 2 units per request per blood type in first pass
            allocate = min(remaining, available, 2)
            
            if allocate > 0:
                allocation_matrix[orig_idx][donor_idx] += allocate
                inventories_copy[donor] -= allocate
                if donor == 'O':
                    available_o -= allocate
                remaining -= allocate

    # ─── SECOND PASS: Fulfill remaining high-priority requests fully ────────
    for orig_idx, quantity, urgency, wait_time, blood_type in active_reqs:
        if orig_idx >= max_requests or quantity <= 0:
            continue

        # Check if request is still partially unfulfilled
        current_alloc = sum(allocation_matrix[orig_idx])
        if current_alloc >= quantity:
            continue  # Already fully allocated in first pass

        compatible = get_compatible_donors(blood_type)
        remaining = quantity - current_alloc

        for donor in compatible:
            if remaining <= 0:
                break
            
            donor_idx = blood_types.index(donor)
            
            if donor == 'O':
                available = available_o
            else:
                available = inventories_copy[donor]
            
            allocate = min(remaining, available)
            
            if allocate > 0:
                allocation_matrix[orig_idx][donor_idx] += allocate
                inventories_copy[donor] -= allocate
                if donor == 'O':
                    available_o -= allocate
                remaining -= allocate

        # Fallback: use O-type reserve for critical unfulfilled requests
        if remaining > 0 and urgency == 2 and o_reserve > 0:
            o_idx = blood_types.index('O')
            allocate = min(remaining, o_reserve)
            if allocate > 0:
                allocation_matrix[orig_idx][o_idx] += allocate
                inventories_copy['O'] -= allocate
                o_reserve -= allocate

    # ─── ZERO-ACTION FIX ─────────────────────────────────────────────────────
    # If no allocation made but inventory and requests exist, allocate 1 unit to top priority
    total_allocation = sum(sum(row) for row in allocation_matrix)
    if total_allocation == 0 and sum(inventories_copy.values()) > 0 and active_reqs:
        top_priority_idx, top_qty, top_urgency, top_wait, top_blood = active_reqs[0]
        if top_priority_idx < max_requests:
            compatible = get_compatible_donors(top_blood)
            for donor in compatible:
                donor_idx = blood_types.index(donor)
                if inventories_copy[donor] > 0:
                    allocation_matrix[top_priority_idx][donor_idx] = 1
                    break

    # Flatten (row-major)
    flattened = []
    for row in allocation_matrix:
        flattened.extend(row)

    return flattened


def get_compatible_donors(recipient_type: str) -> List[str]:
    """
    Return compatible donor blood types for a given recipient,
    ordered from most-preferred (exact match) to least-preferred (universal).

    Preference order keeps O-negative stock from being exhausted too quickly:
    exact match first, then O as last resort.
    """
    compatibility = {
        'O':  ['O'],
        'A':  ['A', 'O'],        # prefer A stock, fall back to O
        'B':  ['B', 'O'],        # prefer B stock, fall back to O
        'AB': ['AB', 'A', 'B', 'O'],  # prefer AB, then specific, then O
    }
    return compatibility.get(recipient_type, ['O'])