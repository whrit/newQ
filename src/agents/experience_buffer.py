# src/agents/experience_buffer.py
import numpy as np
from typing import Tuple, List
import torch
from collections import namedtuple
import random

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class SegmentTree:
    """
    Segment tree data structure for efficient priority updates and sampling
    """
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0
        
    def _propagate(self, idx: int, change: float):
        """Propagate change up through tree"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
            
    def _retrieve(self, idx: int, s: float) -> int:
        """Find sample on leaf node"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
            
    def total(self) -> float:
        """Get total priority"""
        return self.tree[0]
        
    def add(self, priority: float, data: object):
        """Add new data with priority"""
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1
            
    def update(self, idx: int, priority: float):
        """Update priority"""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
        
    def get(self, s: float) -> Tuple[int, float, object]:
        """Get element based on priority"""
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer using Segment Tree
    """
    def __init__(self, capacity: int, batch_size: int, device: str,
                 alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        
        self.tree = SegmentTree(capacity)
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """
        Add experience with maximum priority for new experiences
        """
        experience = Experience(state, action, reward, next_state, done)
        # New experiences get maximum priority
        self.tree.add(self.max_priority ** self.alpha, experience)
        
    def sample(self, batch_size: int) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, List[int]]:
        """
        Sample batch of experiences based on priorities
        """
        batch = []
        indices = []
        priorities = []
        total_priority = self.tree.total()
        
        # Calculate segment size
        segment = total_priority / batch_size
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # Sample uniformly from segment
            s = random.uniform(a, b)
            idx, priority, experience = self.tree.get(s)
            
            indices.append(idx)
            priorities.append(priority)
            batch.append(experience)
            
        # Calculate importance sampling weights
        samples_priority = np.array(priorities)
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate weights
        prob = samples_priority / total_priority
        weights = (self.capacity * prob) ** -self.beta
        weights = weights / weights.max()
        
        # Unpack batch
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])
        
        return (states, actions, rewards, next_states, dones), weights, indices
        
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled experiences"""
        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive and not zero
            priority = max(1e-6, priority)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority ** self.alpha)
            
    def __len__(self) -> int:
        """Return current buffer size"""
        return self.tree.n_entries