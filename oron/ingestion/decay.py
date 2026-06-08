import math
import time
from typing import Dict, Any

class DecayManager:
    """
    Manages memory decay and salience scoring.
    """
    def __init__(self, halflife_days: float = 30):
        self.halflife_seconds = halflife_days * 24 * 60 * 60

    def calculate_episodic_salience(self, timestamp: float, importance: float) -> float:
        """
        Exponential decay for episodic memories.
        Salience = Importance * e^(-lambda * delta_t)
        """
        delta_t = time.time() - timestamp
        decay_constant = math.log(2) / self.halflife_seconds
        salience = importance * math.exp(-decay_constant * delta_t)
        return max(salience, 0.0)

    def calculate_semantic_salience(self, confidence: float) -> float:
        """
        Semantic memories have near-zero decay.
        """
        return confidence

    def calculate_procedural_salience(self, use_count: int, last_used: float) -> float:
        """
        Usage-gated decay for procedural memories.
        Boosted by frequency, slightly decayed by idle time.
        """
        delta_t = time.time() - last_used
        decay_constant = math.log(2) / (self.halflife_seconds * 2) # Slower decay
        usage_boost = math.log1p(use_count) / 5.0 # Logarithmic boost
        salience = (0.8 + usage_boost) * math.exp(-decay_constant * delta_t)
        return min(salience, 1.0)
