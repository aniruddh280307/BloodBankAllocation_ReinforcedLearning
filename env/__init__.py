"""Blood Bank Allocation RL Environment."""

# Use lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'BloodBankEnv':
        from .environment import BloodBankEnv
        return BloodBankEnv
    elif name == 'Observation':
        from .models import Observation
        return Observation
    elif name == 'Action':
        from .models import Action
        return Action
    elif name == 'Reward':
        from .models import Reward
        return Reward
    elif name == 'get_grader':
        from .graders import get_grader
        return get_grader
    elif name == 'greedy_policy':
        from .policy import greedy_policy
        return greedy_policy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    'BloodBankEnv',
    'Observation', 'Action', 'Reward',
    'get_grader',
    'greedy_policy'
]
