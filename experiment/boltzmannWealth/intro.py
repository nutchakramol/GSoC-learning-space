import numpy as np
import mesa
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MoneyAgent(mesa.Agent):
    
    def __init__(self,model):
        super().__init__(model)
        self.wealth = 1
        
    # say hi base on how hard to divide by 2
    def say_hi(self):
        if self.unique_id%2==0:
            print(f"eyh YO, I'm agent {self.unique_id} and I have {self.wealth} dollars")
        else:
            print(f"hi Mate, I'm agent {self.unique_id} and I have {self.wealth} dollars")
            
    def trade(self):
        if self.wealth>0:
            otherAgent = self.random.choice(self.model.agents)
            if otherAgent is not None:
                self.wealth-=1
                otherAgent.wealth+=1
        
        
class MoneyModel(mesa.Model):
    # n is population size
    def __init__(self,n,rng=None):
        super().__init__(rng=rng)
        self.num_agents = n
        MoneyAgent.create_agents(model=self,n=n)
        
    def step(self):
        self.agents.shuffle_do("trade")
#        self.agents.shuffle_do("say_hi")
        
#        self.agents.do("say_hi")
        
        
starter = MoneyModel(n=1000)
for _ in range(100):
    starter.step()
    
agent_wealth = [a.wealth for a in starter.agents]
# Create a histogram with seaborn
g = sns.histplot(agent_wealth, discrete=True)
g.set(
    title="Wealth distribution", xlabel="Wealth", ylabel="number of agents"
);  # The semicolon is just to avoid printing the object representation
plt.show()

all_wealth = []
# This runs the model 100 times, each model executing 30 steps.
"""
for _ in range(100):
    # Run the model
    starter = MoneyModel(10)
    starter.run_for(30)

    # Store the results
    for agent in starter.agents:
        all_wealth.append(agent.wealth)

# Use seaborn
g = sns.histplot(all_wealth, discrete=True)
g.set(title="Wealth distribution", xlabel="Wealth", ylabel="number of agents");
plt.show()
"""
        
