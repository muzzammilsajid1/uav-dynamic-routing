import math

# Authoritative V2 Action Mapping
# Format: { action_index: {"delta": (row_delta, col_delta), "name": name, "type": type, "cost": cost} }

AUTHORITATIVE_ACTION_MAPPING = {
    0: {"delta": (-1, 0), "name": "N", "type": "orthogonal", "cost": 1.0},
    1: {"delta": (1, 0), "name": "S", "type": "orthogonal", "cost": 1.0},
    2: {"delta": (0, -1), "name": "W", "type": "orthogonal", "cost": 1.0},
    3: {"delta": (0, 1), "name": "E", "type": "orthogonal", "cost": 1.0},
    4: {"delta": (-1, -1), "name": "NW", "type": "diagonal", "cost": math.sqrt(2)},
    5: {"delta": (-1, 1), "name": "NE", "type": "diagonal", "cost": math.sqrt(2)},
    6: {"delta": (1, -1), "name": "SW", "type": "diagonal", "cost": math.sqrt(2)},
    7: {"delta": (1, 1), "name": "SE", "type": "diagonal", "cost": math.sqrt(2)},
    8: {"delta": (0, 0), "name": "HOVER", "type": "hover", "cost": 1.0},
}
