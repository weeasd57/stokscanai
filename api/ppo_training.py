import os
import time
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable

class StockTradingEnv(gym.Env):
    """A stock trading environment for gymnasium"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df: pd.DataFrame, initial_balance=10000, reward_mode='pnl', max_steps=None):
        super(StockTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.reward_mode = reward_mode
        # Ensure max_steps doesn't exceed the dataframe length
        if max_steps is not None:
            self.max_steps = min(max_steps, len(self.df) - 1)
        else:
            self.max_steps = len(self.df) - 1
        
        # Action space: 0 = Hold, 1 = Buy (Long), 2 = Sell (Close/Short)
        self.action_space = spaces.Discrete(3)

        # Observation space: 11 features
        # Assuming df has: close, rsi, macd, macd_signal, sma_50, sma_200, 
        # dist_high, dist_low, rsi_diff, vol_sma, day_of_week
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.position = 0  # 1 for Long, -1 for Short, 0 for None
        self.entry_price = 0
        self.current_step = 0
        self.trades = []
        self.history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Ensure current_step does not exceed the last valid index
        # This check is crucial if the episode ends and _get_observation is called for the final state
        # before reset, or if max_steps is set such that current_step could go out of bounds.
        if self.current_step >= len(self.df):
            # This should ideally not happen if done is handled correctly,
            # but as a safeguard, return the last valid observation.
            obs = self.df.iloc[len(self.df) - 1].values
        else:
            obs = self.df.iloc[self.current_step].values
        # Ensure it's 11 features and float32
        return obs.astype(np.float32)

    def step(self, action):
        # We process the action on the CURRENT step's price
        # Get current price - be robust with casing
        if 'close' in self.df.columns:
            current_price = self.df.iloc[self.current_step]['close']
        elif 'Close' in self.df.columns:
            current_price = self.df.iloc[self.current_step]['Close']
        else:
            current_price = self.df.iloc[self.current_step, 0]
        
        # Move to next step
        self.current_step += 1
        
        # Execute trade
        reward = 0
        if action == 1: # Buy / Long
            if self.position <= 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 2: # Sell / Close
            if self.position > 0:
                # Calculate PnL
                pnl = (current_price - self.entry_price) / self.entry_price
                self.balance *= (1 + pnl)
                self.position = 0
                self.entry_price = 0
        
        self.net_worth = self.balance
        if self.position > 0:
            unrealized_pnl = (current_price - self.entry_price) / self.entry_price
            self.net_worth = self.balance * (1 + unrealized_pnl)

        # Reward calculation
        if self.reward_mode == 'pnl':
            reward = (self.net_worth - self.initial_balance) / self.initial_balance
        elif self.reward_mode == 'sharpe':
            # Simplified step-wise reward
            reward = (self.net_worth / self.initial_balance) - 1.0
        
        done = self.current_step >= self.max_steps
        truncated = False
        
        return self._get_observation(), reward, done, truncated, {}

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, progress_cb=None, verbose=0):
        super(ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_cb = progress_cb
        self.last_update = 0

    def _on_step(self) -> bool:
        if self.n_calls % 100 == 0:
            # Update metrics
            policy_loss = self.locals.get('policy_gradient_loss', 0)
            value_loss = self.locals.get('value_loss', 0)
            
            # This is hard to get directly from step, often pooled from logger
            mean_reward = 0
            if hasattr(self.model, 'ep_info_buffer') and self.model.ep_info_buffer:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])

            if self.progress_cb:
                self.progress_cb({
                    "phase": "training",
                    "message": f"Step {self.n_calls}/{self.total_timesteps}",
                    "stats": {
                        "iteration": self.n_calls,
                        "policy_loss": float(policy_loss),
                        "value_loss": float(value_loss),
                        "ep_rew_mean": float(mean_reward)
                    }
                })
        return True

def train_ppo(
    exchange: str,
    df: pd.DataFrame,
    hyperparams: Dict[str, Any],
    progress_cb: Optional[Callable] = None
):
    """
    Trains a PPO model on the given data.
    """
    try:
        # 1. Setup Environment
        initial_balance = hyperparams.get('initial_balance', hyperparams.get('env', {}).get('initialBalance', 10000))
        reward_mode = hyperparams.get('reward_mode', hyperparams.get('env', {}).get('rewardMode', 'pnl'))
        max_steps = hyperparams.get('max_steps', hyperparams.get('env', {}).get('maxSteps', 1000))
        
        env = StockTradingEnv(
            df, 
            initial_balance=initial_balance,
            reward_mode=reward_mode,
            max_steps=max_steps
        )

        # 2. Setup Model
        model_name = hyperparams.get('model_name', hyperparams.get('modelName', f"PPO_{exchange}_{int(time.time())}"))
        net_arch = hyperparams.get('net_arch', hyperparams.get('networkArchitecture', [64, 64]))
        
        # If net_arch is a string (legacy), map it
        if isinstance(net_arch, str):
            arch_map = {
                'small': [32, 32],
                'medium': [64, 64],
                'large': [128, 128]
            }
            net_arch = arch_map.get(net_arch, [64, 64])

        policy_kwargs = dict(net_arch=dict(pi=net_arch, vf=net_arch))

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=hyperparams.get('learning_rate', hyperparams.get('learningRate', 3e-4)),
            n_steps=hyperparams.get('n_steps', hyperparams.get('nSteps', 2048)),
            batch_size=hyperparams.get('batch_size', hyperparams.get('batchSize', 64)),
            n_epochs=hyperparams.get('n_epochs', hyperparams.get('nEpochs', 10)),
            gamma=hyperparams.get('gamma', 0.99),
            clip_range=hyperparams.get('clip_range', hyperparams.get('clipRange', 0.2)),
            ent_coef=hyperparams.get('ent_coef', hyperparams.get('entCoef', 0.0)),
            vf_coef=hyperparams.get('vf_coef', hyperparams.get('vfCoef', 0.5)),
            policy_kwargs=policy_kwargs,
            verbose=1
        )

        # 3. Train
        total_timesteps = hyperparams.get('total_timesteps', hyperparams.get('totalTimesteps', 50000))
        callback = ProgressCallback(total_timesteps, progress_cb)
        
        model.learn(total_timesteps=total_timesteps, callback=callback)

        # 4. Save
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models", "ppo")
        os.makedirs(models_dir, exist_ok=True)
        
        save_path = os.path.join(models_dir, f"{model_name}.zip")
        model.save(save_path)
        
        if progress_cb:
            progress_cb({
                "phase": "completed",
                "message": f"Training completed. Model saved as {model_name}.zip"
            })
            
        return save_path

    except Exception as e:
        if progress_cb:
            progress_cb({
                "phase": "error",
                "message": f"Training failed: {str(e)}"
            })
        raise e
