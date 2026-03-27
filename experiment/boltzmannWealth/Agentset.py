import mesa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

class MoneyAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.wealth = 1
        
        # Each agent gets its own random ethnicity
        ethnicities = ["white", "black", "asian", "hispanic"]
        self.ethnicity = self.random.choice(ethnicities)

    def exchange(self):
        if self.wealth > 0:
            otherAgent = self.random.choice(self.model.agents)
            if otherAgent is not None:
                self.wealth -= 1
                otherAgent.wealth += 1


class MoneyModel(mesa.Model):
    def __init__(self, n=100):
        super().__init__()
        
        # Create n agents (each assigns its own ethnicity)
        MoneyAgent.create_agents(model=self, n=n)

    def step(self):
        self.agents.shuffle_do("exchange")
        
def wealth_bracket(agent):
    if agent.wealth == 0:
        return "broke"
    elif agent.wealth <= 2:
        return "modest"
    else:
        return "wealthy"


# Run model
model = MoneyModel(n=50)
model.run_for(30)

print(f"Total agents: {len(model.agents)}")

# Print agents
for agent in model.agents.select(at_most=50):
    print(
        f"Agent {agent.unique_id}: wealth={agent.wealth}, ethnicity={agent.ethnicity}"
    )
print("     --------------------------")
#compare wealth distri to ethnicity

wealth_by_ethnicity = {}

for agent in model.agents:
    wealth_by_ethnicity.setdefault(agent.ethnicity, []).append(agent.wealth)

for eth, wealths in wealth_by_ethnicity.items():
    print(f"{eth}: avg wealth = {np.mean(wealths):.2f}")
    
print("     --------------------------")

wealthANDeth= model.agents.get(["wealth","ethnicity"])
print("first 5 agent(wealth, ethnicity) ")
for value in wealthANDeth[:5]:
    print(f"{value}")

print("     ---------- group by ---------")

brackets = model.agents.groupby(wealth_bracket)
print("Agents per wealth bracket:")
for bracket, group in brackets:
    print(bracket, len(group))
    
print("=== Model Summary After 50 Steps ===\n")

min_w, max_w, avg_w, total_w = model.agents.agg("wealth", [min, max, np.mean, sum])
print(f"Agents: {len(model.agents)}")
print(
    f"Total wealth: {total_w} (conserved: {'yes' if total_w == len(model.agents) else 'no, subsidy applied'})"
)
print(f"Wealth range: {min_w} to {max_w}, mean: {avg_w:.2f}\n")

print("By ethnicity:")
for ethnicity, group in model.agents.groupby("ethnicity"):
    count = len(group)
    avg = group.agg("wealth", np.mean)
    broke = len(group.select(lambda a: a.wealth == 0))
    print(
        f"  {ethnicity:6s}: {count:3d} agents, avg wealth = {avg:.2f}, broke = {broke}"
    )

print("\nWealth brackets:")
for bracket, group in brackets:
    print(f"  {bracket:8s}: {len(group)} agents")
    
# Collect data for plotting
data = []
for agent in model.agents:
    data.append({"wealth": agent.wealth, "ethnicity": agent.ethnicity})
df = pd.DataFrame(data)

palette = {
    "white": "gray",
    "black": "black",
    "asian": "blue",
    "hispanic": "orange"
}
g = sns.histplot(data=df, x="wealth", hue="ethnicity", discrete=True, palette=palette)
g.set(
    title="Wealth distribution by ethnicity", xlabel="Wealth", ylabel="Number of agents"
)
plt.show()