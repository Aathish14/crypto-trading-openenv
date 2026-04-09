"""
OpenEnv Graders for the Cryptocurrency Trading Environment.
Ensures Phase 2 compliance with scores strictly between 0 and 1.
"""

import numpy as np

class CryptoTradingGrader:
    """Base grader for crypto trading tasks."""
    
    def __init__(self, success_threshold: float):
        self.success_threshold = success_threshold

    def grade(self, env) -> float:
        """
        Grades the agent's performance based on portfolio return.
        Returns a score strictly between 0.01 and 0.99.
        """
        # Ensure we can access the environment state
        try:
            # Extract metrics from the environment's info or state
            # If the episode is finished, env.portfolio_value should be final
            initial_balance = getattr(env, 'initial_balance', 10000)
            current_value = getattr(env, 'portfolio_value', initial_balance)
            
            # Calculate total return percentage
            total_return = (current_value - initial_balance) / initial_balance
            
            # Calculate how close we are to the success threshold
            # Progress is 0 when return <= 0, and 1 when return >= threshold
            progress = max(0.0, total_return) / self.success_threshold
            
            # Map [0, inf) to [0.01, 0.99] using a small epsilon and capping
            # We never want to return 0.0 or 1.0 per Phase 2 requirements
            score = 0.01 + 0.98 * min(1.0, progress)
            
            # Final safety check for strict range (0, 1)
            return float(np.clip(score, 0.001, 0.999))
            
        except Exception as e:
            print(f"Grading error: {e}")
            return 0.05 # Minimum "attempt" score

class BasicTradingGrader(CryptoTradingGrader):
    def __init__(self):
        super().__init__(success_threshold=0.10) # 10% return for Easy

class IntermediateTradingGrader(CryptoTradingGrader):
    def __init__(self):
        super().__init__(success_threshold=0.25) # 25% return for Medium

class AdvancedTradingGrader(CryptoTradingGrader):
    def __init__(self):
        super().__init__(success_threshold=0.50) # 50% return for Hard
