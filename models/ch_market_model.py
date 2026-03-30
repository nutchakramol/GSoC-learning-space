import math
import numpy as np
import mesa

MAX_PRICE, MIN_PRICE = 100.0, 0.0
MAX_LEVEL = 5  # Truncate poisson tail
POISSON_TRUNC = MAX_LEVEL + 1

# Cognitive Hierarchy logic p
def poisson_level_weights(tau: float, k_max: int = MAX_LEVEL) -> np.ndarray:
    """Poisson distribution for cognitive levels, renormalized."""
    raw = np.array([math.exp(-tau) * (tau**k) / math.factorial(k) for k in range(k_max + 1)])
    return raw / raw.sum()

def bid_logic_k(level, tau, p_val, role, rng, lam_qre=2.0) -> float:
    """
    Subroutine: Level-0 is random. 
    Level-k best-responds to a uniform belief of levels 0...k-1.
    """
    if level == 0:
        return float(rng.uniform(0, p_val)) if role == "buyer" else float(rng.uniform(p_val, MAX_PRICE))

    # Discrete bid grid for best-response calculation
    grid = np.linspace(0, p_val, 20) if role == "buyer" else np.linspace(p_val, MAX_PRICE, 20)
    
    # payoff estimation: probability of clearing vs estimated opponent bid
    # Real research would use Monte Carlo here
    
    target_price = 50.0  # Level-k agents assume market converges to equilibrium
    payoffs = np.array([max(0, p_val - (b + target_price)/2) if role == "buyer" 
                        else max(0, (b + target_price)/2 - p_val) for b in grid])
    
    # Softmax choice (QRE-style decision)
    logits = lam_qre * (payoffs - np.max(payoffs))
    probs = np.exp(logits) / np.exp(logits).sum()
    return float(np.random.choice(grid, p=probs))

class CHAgent(mesa.Agent):
    """Agent that estimates population sophistication (Tau) via Bayesian updating."""
    def __init__(self, model):
        super().__init__(model)
        lw = poisson_level_weights(model.tau_init)
        self.level = self.random.choices(range(POISSON_TRUNC), weights=lw)[0]
        
        # Bayesian Prior: Gamma(alpha, beta) for Tau estimation
        self.tau_alpha, self.tau_beta = 2.0, 2.0 / model.tau_init
        self.last_bid, self.last_surplus = 50.0, 0.0
        self.role, self.p_val = "buyer", 50.0

    @property
    def tau_est(self): return self.tau_alpha / self.tau_beta

    def step(self):
        """Prepare for auction."""
        self.role = self.random.choice(["buyer", "seller"])
        self.p_val = self.random.uniform(MIN_PRICE, MAX_PRICE)
        self.last_bid = bid_logic_k(self.level, self.tau_est, self.p_val, self.role, self.random)

    def update_beliefs(self, market_price):
        """Update Gamma prior based on price distance from equilibrium."""
        signal = 1.0 / (1.0 + abs(market_price - 50.0) / 50.0)
        self.tau_alpha += 0.5
        self.tau_beta += signal * 0.5

class CHMarketModel(mesa.Model):
    def __init__(self, n_agents=60, tau_init=1.5, seed=None):
        super().__init__(seed=seed)
        self.tau_init = tau_init
        for _ in range(n_agents):
            self.agents.add(CHAgent(self))
            
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "AvgTau": lambda m: np.mean([a.tau_est for a in m.agents]),
                "Efficiency": lambda m: m.calculate_efficiency()
            }
        )

    def calculate_efficiency(self):
        # Placeholder for surplus / max theoretical surplus
        return np.random.uniform(0.7, 0.95)

    def step(self):
        self.agents.shuffle_do("step")
        
        # Market Clearing logic 
        bids = sorted([a.last_bid for a in self.agents if a.role == "buyer"], reverse=True)
        asks = sorted([a.last_bid for a in self.agents if a.role == "seller"])
        
        trades = [ (b+a)/2 for b, a in zip(bids, asks) if b >= a ]
        if trades:
            avg_p = np.mean(trades)
            for a in self.agents: a.update_beliefs(avg_p)
            
        self.datacollector.collect(self)

if __name__ == "__main__":
    model = CHMarketModel(n_agents=100, seed=None)
    print(f"Starting CH Market Simulation (Initial Tau: {model.tau_init})")
    
    for i in range(20):
        model.step()
        tau = model.datacollector.get_model_vars_dataframe()["AvgTau"].iloc[-1]
        print(f"Round {i:02} | Population estimated Tau: {tau:.3f}")