"""Baseline policy: greedy allocation strategy."""
import random
from typing import List


def greedy_policy(obs) -> List[int]:
    """
    Aggressive greedy allocation strategy:
    1. Sort requests by urgency (high > medium > low) and wait time
    2. For each request, allocate the FULL requested quantity from compatible types
    3. Never scale down — partial fulfillment is always better than no fulfillment

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

    # ── Dead-zone guard ──────────────────────────────────────────────────────
    # If there is literally nothing in stock, no allocation is possible.
    # Return zeros immediately rather than wasting cycles.
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

    # ── Dead-zone guard (no pending requests) ────────────────────────────────
    if not active_reqs:
        return [0] * (max_requests * 4)

    # Sort: urgency DESC, then wait_time DESC
    active_reqs.sort(key=lambda x: (-x[2], -x[3]))

    # Build allocation matrix (5 × 4)
    allocation_matrix  = [[0] * 4 for _ in range(max_requests)]
    inventories_copy   = inventories.copy()

    for orig_idx, quantity, urgency, wait_time, blood_type in active_reqs:
        if orig_idx >= max_requests:
            break
        if quantity <= 0:
            continue

        compatible = get_compatible_donors(blood_type)

        # ── KEY FIX: always try to fulfil the FULL quantity ──────────────────
        # Never scale down.  Partial allocation still earns positive reward and
        # avoids the penalty that unmet requests accumulate every step.
        remaining = quantity

        for donor in compatible:
            if remaining <= 0:
                break
            donor_idx = blood_types.index(donor)
            available = inventories_copy[donor]
            alloc     = min(remaining, available)
            if alloc > 0:
                allocation_matrix[orig_idx][donor_idx]  = alloc
                inventories_copy[donor]                -= alloc
                remaining                              -= alloc

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


def random_policy(obs) -> List[int]:
    """Random allocation (baseline for comparison)."""
    return [random.randint(0, 2) for _ in range(20)]