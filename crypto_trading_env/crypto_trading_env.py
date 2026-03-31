import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, Optional
import yfinance as yf
from datetime import datetime, timedelta


class CryptoTradingEnv(gym.Env):
    """
    A cryptocurrency trading environment for reinforcement learning.

    The agent can trade multiple cryptocurrencies (BTC, ETH, BNB, XRP, LTC)
    using 5-minute candle data. Actions represent position sizing decisions
    (25%, 50%, 75%, 100% of available balance) for each asset.
    """

    def __init__(
        self,
        symbols=["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD"],
        timeframe="5m",
        initial_balance=10000,
        transaction_fee=0.001,
        lookback_window=10,
    ):
        """
        Initialize the cryptocurrency trading environment.

        Args:
            symbols: List of cryptocurrency trading pairs
            timeframe: Data timeframe (default: '5m' for 5-minute candles)
            initial_balance: Starting portfolio balance in USD
            transaction_fee: Fee percentage per trade (default: 0.001 = 0.1%)
            lookback_window: Number of past observations to include in state
        """
        super(CryptoTradingEnv, self).__init__()

        self.symbols = symbols
        self.timeframe = timeframe
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee
        self.lookback_window = lookback_window

        # Action space: For each asset, we can choose:
        # 0: Hold (0% of balance)
        # 1: Buy 25% of balance
        # 2: Buy 50% of balance
        # 3: Buy 75% of balance
        # 4: Buy 100% of balance
        # 5: Sell 25% of balance
        # 6: Sell 50% of balance
        # 7: Sell 75% of balance
        # 8: Sell 100% of balance
        # Actually, let's simplify to: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy75%, 4=Buy100%
        # And selling is handled by having negative positions or separate logic
        # Let's use: 0=Hold, 1=Buy25%, 2=Buy50%, 3=Buy75%, 4=Buy100%
        # For selling, we'll allow selling portions of current holdings
        # Actually, let's reconsider: Discrete position sizing as specified
        # 0: Hold (0% trade)
        # 1: Buy 25% of balance
        # 2: Buy 50% of balance
        # 3: Buy 75% of balance
        # 4: Buy 100% of balance

        # But we also need to handle selling. Let's make it:
        # 0: Hold
        # 1-4: Buy 25%, 50%, 75%, 100%
        # 5-8: Sell 25%, 50%, 75%, 100% of current holdings

        # Actually, let's stick to the specification: "Discrete: Position sizing" with "25%/50%/75%/100% of balance"
        # This likely means we can trade those percentages of our balance
        # So for each asset, we have 5 actions: 0 (hold), 1 (25% buy), 2 (50% buy), 3 (75% buy), 4 (100% buy)
        # And we can sell by having the ability to go short or by selling existing holdings

        # Let's implement it as: For each asset, choose action from [0, 1, 2, 3, 4] where:
        # 0 = Hold (0% of balance)
        # 1 = Buy 25% of balance
        # 2 = Buy 50% of balance
        # 3 = Buy 75% of balance
        # 4 = Buy 100% of balance
        # Selling is done by choosing to sell existing holdings (separate logic)

        # Actually, let's make it simpler and more realistic:
        # Action for each asset: integer from 0 to 4 representing:
        # 0: Hold (no change)
        # 1: Buy 25% of available balance
        # 2: Buy 50% of available balance
        # 3: Buy 75% of available balance
        # 4: Buy 100% of available balance
        # To sell, we would need to have the ability to sell portions of holdings
        # Let's revise: Action represents the target portfolio percentage for each asset
        # But that's continuous... Let me re-read the spec.

        # Re-reading: "Discrete: Position sizing" with "25%/50%/75%/100% of balance"
        # I think this means at each step, for each asset, the agent can choose to:
        # - Hold current position
        # - Buy additional 25%, 50%, 75%, or 100% of balance worth
        # - Sell 25%, 50%, 75%, or 100% of current holdings

        # Let's go with 9 actions per asset:
        # 0: Hold
        # 1: Buy 25% of balance
        # 2: Buy 50% of balance
        # 3: Buy 75% of balance
        # 4: Buy 100% of balance
        # 5: Sell 25% of holdings
        # 6: Sell 50% of holdings
        # 7: Sell 75% of holdings
        # 8: Sell 100% of holdings

        self.n_actions_per_asset = 9
        self.n_assets = len(symbols)
        self.action_space = spaces.MultiDiscrete(
            [self.n_actions_per_asset] * self.n_assets
        )

        # Observation space:
        # For each asset: OHLCV data (Open, High, Low, Close, Volume) for lookback_window periods
        # Plus portfolio information: balance and holdings for each asset
        # So: (lookback_window * 5 * n_assets) + 1 (balance) + n_assets (holdings)
        obs_size = (lookback_window * 5 * self.n_assets) + 1 + self.n_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        # Load data
        self.data = self._load_data()
        self.current_step = 0
        self.max_steps = len(self.data) - lookback_window - 1

        # Portfolio state
        self.balance = 0
        self.holdings = np.zeros(self.n_assets)  # Amount of each asset held
        self.portfolio_value = 0

    def _load_data(self):
        """Load historical price data for all symbols."""
        # For now, we'll generate synthetic data since we don't want to rely on external APIs in this example
        # In a real implementation, you would load actual historical data
        print("Loading synthetic data for demonstration...")

        # Generate synthetic price data
        n_points = 1000  # Number of 5-minute candles
        data = {}

        for symbol in self.symbols:
            # Generate realistic price movements
            base_price = {
                "BTC-USD": 50000,
                "ETH-USD": 3000,
                "BNB-USD": 400,
                "XRP-USD": 0.5,
                "LTC-USD": 100,
            }.get(symbol, 100)

            # Generate random walk with volatility
            returns = np.random.normal(
                0, 0.02, n_points
            )  # 2% volatility per 5-min candle
            prices = base_price * np.exp(np.cumsum(returns))

            # Generate OHLCV data
            opens = prices * (1 + np.random.normal(0, 0.005, n_points))
            highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_points)))
            lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_points)))
            closes = prices
            volumes = np.random.lognormal(
                10, 1, n_points
            )  # Log-normal volume distribution

            # Ensure high >= low and high >= open,close and low <= open,close
            highs = np.maximum(highs, np.maximum(opens, closes))
            lows = np.minimum(lows, np.minimum(opens, closes))

            data[symbol] = pd.DataFrame(
                {
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes,
                }
            )

        # Combine data into a single DataFrame with MultiIndex columns
        combined_data = pd.concat(data, axis=1)
        return combined_data

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Reset portfolio
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.n_assets)  # Start with no holdings
        self.current_step = self.lookback_window  # Start after lookback window

        # Calculate initial portfolio value
        self.portfolio_value = self._calculate_portfolio_value()

        # Get initial observation
        observation = self._get_observation()

        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "holdings": self.holdings.copy(),
        }

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step within the environment.

        Args:
            action: Array of actions for each asset (0-8 as defined)

        Returns:
            observation: New state observation
            reward: Reward for the action taken
            terminated: Whether episode is done
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Execute trades based on actions
        self._execute_trades(action)

        # Move to next time step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False  # We don't truncate episodes in this implementation

        # Calculate new portfolio value and reward
        old_portfolio_value = self.portfolio_value
        self.portfolio_value = self._calculate_portfolio_value()
        reward = self.portfolio_value - old_portfolio_value  # Simple profit/loss reward

        # Get new observation
        observation = self._get_observation()

        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "holdings": self.holdings.copy(),
            "total_return": (self.portfolio_value - self.initial_balance)
            / self.initial_balance,
        }

        return observation, reward, terminated, truncated, info

    def _execute_trades(self, actions: np.ndarray):
        """Execute trades based on the given actions."""
        current_prices = self._get_current_prices()

        for asset_idx, action in enumerate(actions):
            symbol = self.symbols[asset_idx]
            price = current_prices[asset_idx]

            if action == 0:  # Hold
                continue
            elif 1 <= action <= 4:  # Buy actions
                # Calculate amount to spend (percentage of balance)
                buy_fraction = (action) / 4.0  # 1->0.25, 2->0.5, 3->0.75, 4->1.0
                amount_to_spend = self.balance * buy_fraction

                if amount_to_spend > 0:
                    # Calculate fees
                    fee = amount_to_spend * self.transaction_fee
                    total_cost = amount_to_spend + fee

                    if total_cost <= self.balance:
                        # Execute buy
                        amount_bought = amount_to_spend / price
                        self.balance -= total_cost
                        self.holdings[asset_idx] += amount_bought
            elif 5 <= action <= 8:  # Sell actions
                # Calculate amount to sell (percentage of holdings)
                sell_fraction = (action - 4) / 4.0  # 5->0.25, 6->0.5, 7->0.75, 8->1.0
                amount_to_sell = self.holdings[asset_idx] * sell_fraction

                if amount_to_sell > 0:
                    # Execute sell
                    proceeds = amount_to_sell * price
                    fee = proceeds * self.transaction_fee
                    net_proceeds = proceeds - fee

                    self.balance += net_proceeds
                    self.holdings[asset_idx] -= amount_to_sell

    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all assets."""
        prices = []
        for symbol in self.symbols:
            close_price = self.data.loc[
                self.data.index[self.current_step], (symbol, "close")
            ]
            prices.append(close_price)
        return np.array(prices)

    def _calculate_portfolio_value(self) -> float:
        """Calculate total portfolio value (balance + holdings value)."""
        current_prices = self._get_current_prices()
        holdings_value = np.sum(self.holdings * current_prices)
        return self.balance + holdings_value

    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        obs = []

        # Add historical price data for each asset
        for symbol in self.symbols:
            # Get lookback_window periods of OHLCV data
            start_idx = self.current_step - self.lookback_window
            end_idx = self.current_step

            if start_idx < 0:
                # Pad with zeros if we don't have enough history
                ohlcv_data = np.zeros((self.lookback_window, 5))
                actual_data = self.data.loc[
                    self.data.index[0:end_idx],
                    (symbol, ["open", "high", "low", "close", "volume"]),
                ].values
                ohlcv_data[-len(actual_data) :] = actual_data
            else:
                ohlcv_data = self.data.loc[
                    self.data.index[start_idx:end_idx],
                    (symbol, ["open", "high", "low", "close", "volume"]),
                ].values

            obs.extend(ohlcv_data.flatten())

        # Add portfolio information
        obs.append(self.balance)
        obs.extend(self.holdings)

        return np.array(obs, dtype=np.float32)

    def render(self, mode="human"):
        """Render the environment."""
        if mode == "human":
            print(f"Step: {self.current_step}")
            print(f"Balance: ${self.balance:.2f}")
            print(f"Holdings: {self.holdings}")
            print(f"Portfolio Value: ${self.portfolio_value:.2f}")
            print(
                f"Total Return: {(self.portfolio_value - self.initial_balance) / self.initial_balance * 100:.2f}%"
            )
            print("-" * 50)

    def state(self):
        """Get the current environment state."""
        # Return a dictionary with the current state information
        return {
            "current_step": self.current_step,
            "balance": self.balance,
            "holdings": self.holdings.copy(),
            "portfolio_value": self.portfolio_value,
            "symbols": self.symbols.copy(),
        }


# Example usage
if __name__ == "__main__":
    # Create environment
    env = CryptoTradingEnv()

    # Test reset
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Test a few steps
    for i in range(5):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated:
            break

    print("Environment test completed!")
