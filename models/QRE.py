import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ACTIONS = ["Hawk", "Dove", "Bourgeois"]
N_ACTIONS = 3

def build_hdb_payoff(V: float = 4.0, C: float = 6.0) -> np.ndarray:
    """Standard Hawk-Dove-Bourgeois payoff matrix."""
    hh, hd, dh = (V - C) / 2, V, 0.0
    hb, bh = (V - C) / 4 + V / 2, (V - C) / 4
    dd, db, bd, bb = V / 2, V / 4, 3 * V / 4, V / 2
    return np.array([[hh, hd, hb], [dh, dd, db], [bh, bd, bb]])

def logit_qre(payoff_matrix, lam, tol=1e-6, max_iter=400):
    """Fixed-point iteration for Logit QRE."""
    n = payoff_matrix.shape[0]
    sigma = np.ones(n) / n
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
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs)))

class QREAgent(mesa.Agent):
    """Agents evolve their rationality via social imitation."""
    
    def __init__(self, model, lam: float = None):
        super().__init__(model)
        # Use model's mean_lam_init for the distribution
        self.lam = lam if lam is not None else max(0.0, self.random.gauss(self.model.mean_lam_init, 0.5))
        
        self.last_action = 0
        self.last_payoff = 0.0
        self.qre_sigma = np.ones(N_ACTIONS) / N_ACTIONS

    def step(self):
        # Phase 1: Decision making
        self.qre_sigma = logit_qre(self.model.payoff_matrix, self.lam)
        self.last_action = self.model.random.choices([0, 1, 2], weights=self.qre_sigma)[0]

    def advance(self):
        # Phase 2: Interaction and social learning
        # Pick a random partner from the AgentSet
        partner = self.model.random.choice(list(self.model.agents))
        if partner == self: return

        # Calculate payoffs for both
        self.last_payoff = float(self.model.payoff_matrix[self.last_action, partner.last_action])
        
        # Imitation Logic
        if self.model.random.random() < self.model.imitation_prob:
            observer = self.model.random.choice(list(self.model.agents))
            if observer.last_payoff > self.last_payoff:
                noise = self.model.random.gauss(0, self.model.lam_noise)
                self.lam = max(0.0, observer.lam + noise)

class QREModel(mesa.Model):
    def __init__(self, n_agents=80, mean_lam_init=1.5, imitation_prob=0.3, lam_noise=0.1, seed=42):
        super().__init__(seed=seed)
        self.rng = np.random.default_rng(seed)
        self.payoff_matrix = build_hdb_payoff()
        self.mean_lam_init = mean_lam_init
        self.imitation_prob = imitation_prob
        self.lam_noise = lam_noise
        self.step_count = 0 # Track years/generations

        for _ in range(n_agents):
            self.agents.add(QREAgent(self))

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "MeanLambda": lambda m: np.mean([a.lam for a in m.agents]),
                "MeanPayoff": lambda m: np.mean([a.last_payoff for a in m.agents]),
                "PopEntropy": lambda m: entropy(np.mean([a.qre_sigma for a in m.agents], axis=0)),
            }
        )

    def step(self):
        self.agents.shuffle_do("step")
        self.agents.shuffle_do("advance")
        self.datacollector.collect(self)
        self.step_count += 1

        data = self.datacollector.get_model_vars_dataframe().iloc[-1]
        avg_lam = data["MeanLambda"]
        avg_payoff = data["MeanPayoff"]
        
        # state of the world
        rationality_status = "chaotic/random" if avg_lam < 0.5 else "learning" if avg_lam < 2.0 else "highly strategic"
        
        print(f"--- Generation {self.step_count} ---")
        print(f"The population is currently {rationality_status}.")
        print(f"On average, agents are earning {avg_payoff:.2f} points per interaction.")
        print(f"The collective 'intelligence' (Lambda) has reached {avg_lam:.4f}.")
        print("-" * 30)
        
def run_experiment():
    #  how lam noise and imitation prob affect outcome
    params = {
        "n_agents": [50],
        "mean_lam_init": [1.0],
        "imitation_prob": [0.1, 0.5], # Low vs High social pressure
        "lam_noise": [0.05, 0.2],     # Low vs High mutation/noise
        "seed": range(5)              # Run 5 iterations of each for statistical validity
    }

    print("LOG: Starting Batch Simulation...")
    
    results = mesa.batch_run(
        QREModel,
        parameters=params,
        iterations=1, 
        max_steps=100,     #  how long each "world" lives
        number_processes=None, 
        data_collection_period=1,
        display_progress=True,
    )

    df = pd.DataFrame(results)
    
    final_states = df[df.Step == 100]

    # Grouping by Noise and Imitation to see the average final Lambda
    analysis = final_states.groupby(["lam_noise", "imitation_prob"])["MeanLambda"].mean().unstack()
    
    print("\n" + "="*50)
    print("HYPOTHESIS TEST: FINAL MEAN LAMBDA")
    print("="*50)
    print(analysis)
    print("-" * 50)
    
    best_combo = analysis.stack().idxmax()
    print(f"WINNING CONDITION: Noise={best_combo[0]}, Imitation={best_combo[1]}")
    print(f"This environment produced the highest average rationality.")
    
    return df

if __name__ == "__main__":
    df_results = run_experiment()

    pivot_df = df_results[df_results.Step == 100].pivot_table(
        values="MeanLambda", 
        index="lam_noise", 
        columns="imitation_prob", 
        aggfunc="mean"
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", fmt=".3f")
    plt.title("Final Mean Lambda by Environment Condition")
    plt.show()

    best_noise = 0.20
    best_imit = 0.5
    
    sample_run = df_results[
        (np.isclose(df_results["lam_noise"], best_noise)) & 
        (np.isclose(df_results["imitation_prob"], best_imit)) & 
        (df_results["seed"] == 0)
    ].sort_values("Step")

    if not sample_run.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(sample_run['Step'], sample_run['MeanLambda'], color='blue', label="Best Run")
        plt.axhline(y=1.0, color='gray', linestyle='--', label="Initial Lambda")
        plt.title(f"Rationality Growth: Noise={best_noise}, Imitation={best_imit}")
        plt.xlabel("Step")
        plt.ylabel("Mean Lambda")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
    else:
        print("Filter failed again. Available columns are:", df_results.columns)
        print("Unique Noise values:", df_results["lam_noise"].unique())