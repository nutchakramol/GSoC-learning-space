import math
import random 
from dataclasses import dataclass, models
from typing import TYPE_CHECKING
import numpy as np
import mesa

class QREAgent(mesa.Agent):
    """Agents evolve their rationality via social imitation"""
    def __init__(self, model,lam:float= None):
        super().__init__(model)
        if lam is None:
            self.lam = max(0.0, self.random.gauss(self.model.mean_lam_init,0.5))
        else:
            self.lam = lam
            
        self.last_action: int = 0
        self.last_payoff: float = 0.0
        self.qre_sigma: np.ndarray = np.ones(N_ACTIONS) / N_ACTIONS

    def _recompute_qre(self) -> None:
        self.qre_sigma = logit_qre(self.model.payoff_matrix, self.lam)

    def _sample_action(self) -> int:
        return int(self.model.rng.choice(N_ACTIONS, p=self.qre_sigma))

    def step(self) -> None:
        self._recompute_qre()
        self.last_action = self._sample_action()

        # Play against one random opponent
        partner: QREAgent = self.model.random.choice(
            [a for a in self.model.agents if a is not self]
        )
        partner._recompute_qre()
        partner_action = partner._sample_action()

        self.last_payoff = float(
            self.model.payoff_matrix[self.last_action, partner_action]
        )
        partner.last_payoff = float(
            self.model.payoff_matrix[partner_action, self.last_action]
        )

        # observe random agent, copy λ if they did better
        if self.model.random.random() < self.model.imitation_prob:
            observer: QREAgent = self.model.random.choice(self.model.agents)
            if observer is not self and observer.last_payoff > self.last_payoff:
                noise = self.model.random.gauss(0, self.model.lam_noise)
                self.lam = max(0.0, observer.lam + noise)
                
class QREModel(mesa.Model):
    
    def __init__(
        self,
        n_agents: int = 80,
        payoff_matrix: np.ndarray | None = None,
        mean_lam_init: float = 1.5,
        imitation_prob: float = 0.3,
        lam_noise: float = 0.1,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.payoff_matrix: np.ndarray = (
            payoff_matrix if payoff_matrix is not None else build_hdb_payoff()
        )
        self.mean_lam_init = mean_lam_init
        self.imitation_prob = imitation_prob
        self.lam_noise = lam_noise
        self.step_count = 0

        # Compute Nash equilibrium via QRE at very reference point
        self.nash_approx: np.ndarray = logit_qre(self.payoff_matrix, lam=50.0)

        self.agents: list[QREAgent] = [QREAgent(self) for _ in range(n_agents)]

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "MeanLambda": lambda m: float(np.mean([a.lam for a in m.agents])),
                "StdLambda":  lambda m: float(np.std([a.lam for a in m.agents])),
                "MeanPayoff": lambda m: float(np.mean([a.last_payoff for a in m.agents])),
                "PopEntropy": lambda m: float(
                    entropy(np.mean([a.qre_sigma for a in m.agents], axis=0))
                ),
                "FreqHawk":       lambda m: float(np.mean([a.last_action == 0 for a in m.agents])),
                "FreqDove":       lambda m: float(np.mean([a.last_action == 1 for a in m.agents])),
                "FreqBourgeois":  lambda m: float(np.mean([a.last_action == 2 for a in m.agents])),
            },
            agent_reporters={
                "Lambda": "lam",
                "Action": "last_action",
                "Payoff": "last_payoff",
            },
        )
        
    def step(self) -> None:
        """all agents act, then collect data"""
        shuffled = self.agents[:]
        self.random.shuffle(shuffled)
        for agent in shuffled:
            agent.step()
        self.datacollector.collect(self)
        self.step_count += 1

def build_hdb_payoff(V: float = 4.0, C: float = 6.0) -> np.ndarray:
    """Standard Hawk-Dove-Bourgeois payoff matrix"""
    hh = (V - C) / 2
    hd = V
    dh = 0.0
    hb = (V - C) / 4 + V / 2
    bh = (V - C) / 4
    dd = V / 2
    db = V / 4
    bd = 3 * V / 4
    bb = V / 2
    
    return np.array([
        [hh, hd, hb],   # Hawk row
        [dh, dd, db],   # Dove 
        [bh, bd, bb],   # Bourgeois 
    ])

def logit_qre(
    payoff_matrix: np.ndarray,
    lam: float,
    tol: float = 1e-6,
    max_iter: int = 400,
) -> np.ndarray:
    """Fixed-point iteration for Logit QRE."""
    n = payoff_matrix.shape[0]
    sigma = np.ones(n) / n
     # QRE is uniform when λ=0 by definition
    if lam == 0.0: return sigma

    for _ in range(max_iter):
        eu = payoff_matrix @ sigma
 
        logits = lam * eu
        exp_logits = np.exp(logits - np.max(logits))
        new_sigma = exp_logits / exp_logits.sum()
        
        if np.max(np.abs(new_sigma - sigma)) < tol:
            return new_sigma
        sigma = new_sigma
    return sigma

def entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability distribution (nats)."""
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))


ACTIONS = ["Hawk", "Dove", "Bourgeois"]
N_ACTIONS = 3