import mesa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from mesa.time import Priority, Schedule

class MoneyAgent(mesa.Agent):
    """An agent with fixed initial wealth."""
    def __init__(self, model):
        super().__init__(model)
        self.wealth = 1

    def exchange(self):
        if self.wealth > 0:
            other = self.random.choice(self.model.agents)
            other.wealth += 1
            self.wealth -= 1


class MoneyModel(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, n=10):
        super().__init__()
        MoneyAgent.create_agents(model=self, n=n)

    def step(self):
        self.agents.shuffle_do("exchange")
        
class fixedOrder(mesa.Model):
    def __init__(self,n=10):
        super().__init__()
        MoneyAgent.create_agents(model=self,n=n)
    def step(self):
        self.agents.do("exchange")
        
class SimpleModel(mesa.Model):
    def __init__(self):
        super().__init__()
        self.steps = 0

    def step(self):
        print(f"  Step {self.steps} at time {self.time:.1f}")
        
class EconomyModel(mesa.Model):
    """A simple economy where a tax reform happens at a specific time."""

    def __init__(self, n=50):
        super().__init__()
        self.tax_rate = 0.1
        self.events_log = []

        # Create agents with wealth
        for _ in range(n):
            a = mesa.Agent(self)
            a.wealth = 10

        # Schedule a tax reform at time 5.0
        self.schedule_event(self.tax_reform, at=5.0)

        # Schedule a stimulus check 2 time units from now (so at time 2.0)
        self.schedule_event(self.stimulus, after=2.0)

    def tax_reform(self):
        self.tax_rate = 0.25
        self.events_log.append(
            f"t={self.time:.1f}: Tax reform! Rate now {self.tax_rate}"
        )

    def stimulus(self):
        for agent in self.agents:
            agent.wealth += 5
        self.events_log.append(f"t={self.time:.1f}: Stimulus! Everyone gets 5 units")

    def step(self):
        # Simple taxation each step
        for agent in self.agents:
            tax = int(agent.wealth * self.tax_rate)
            agent.wealth -= tax
            
def gini(model):
    x=sorted(model.agents.get("wealth"))
    n=len(x)
    B = sum(xi * (n - i) for i, xi in enumerate(x)) / (n * sum(x))
    return 1 + (1 / n) - 2 * B
        
def tax_agent(agent):
    """Take 10% tax from agents with wealth > 5."""
    if agent.wealth > 5:
        tax = agent.wealth // 10
        agent.wealth -= tax


m = MoneyModel(10)
m.run_for(30)

wealth = m.agents.get("wealth")
print(f"Wealth distribution: {sorted(wealth, reverse=True)}")
print(f"Total wealth: {sum(wealth)} (should be {len(m.agents)})")

fixed_ginis = []
random_ginis = []
for _ in range(50):
    m = fixedOrder(50)
    m.run_for(100)
    fixed_ginis.append(gini(m))

    m = MoneyModel(50)
    m.run_for(100)
    random_ginis.append(gini(m))

print(
    f"Fixed order  — mean Gini: {np.mean(fixed_ginis):.3f} (std: {np.std(fixed_ginis):.3f})"
)
print(
    f"Random order — mean Gini: {np.mean(random_ginis):.3f} (std: {np.std(random_ginis):.3f})"
)

print(f"Max wealth before tax: {m.agents.agg('wealth', max)}")
m.agents.do(tax_agent)
print(f"Max wealth after tax: {m.agents.agg('wealth', max)}")

print("   ------   scheduling --------")

model = SimpleModel()
print("Initial state:")
print(f"  steps={model.steps}, time={model.time}")
print("\nRunning for 3 time units:")
model.run_for(3)
print(f"\nFinal state: steps={model.steps}, time={model.time}")

print(" ======== event scheduling =========")
model2 = EconomyModel(10)
model2.run_for(7)

print("Events that occurred:")
for event in model2.events_log:
    print(f"  {event}")

print(f"\nFinal tax rate: {model2.tax_rate}")
avg_wealth = model2.agents.agg("wealth", np.mean)
print(f"Average wealth: {avg_wealth:.1f}")