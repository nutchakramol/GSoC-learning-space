import mesa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
class Prey(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.energy = 5

    def move(self):
        self.energy -= 1

    def eat(self):
        self.energy += self.random.randint(0, 4)

    def reproduce(self):
        if self.energy > 8:
            self.energy -= 4
            Prey(self.model)  # New prey is automatically registered


class Predator(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.energy = 10
        self.kills = 0

    def move(self):
        self.energy -= 2  # Predators use more energy

    def hunt(self):
        prey_agents = self.model.agents_by_type.get(Prey)
        if prey_agents and len(prey_agents) > 0 and self.energy > 0:
            target = self.random.choice(prey_agents)
            target.remove()
            self.energy += 5
            self.kills += 1


class EcosystemModel(mesa.Model):
    def __init__(self, n_prey=100, n_predators=10):
        super().__init__()
        Prey.create_agents(model=self, n=n_prey)
        Predator.create_agents(model=self, n=n_predators)

    def step(self):
        # Stage 1: All agents move (random order within each type)
        for agent_type in self.agent_types:
            self.agents_by_type[agent_type].shuffle_do("move")

        # Stage 2: Type-specific actions
        if Prey in self.agents_by_type:
            self.agents_by_type[Prey].shuffle_do("eat")
        if Predator in self.agents_by_type:
            self.agents_by_type[Predator].shuffle_do("hunt")

        # Stage 3: Only prey with enough energy reproduce
        if Prey in self.agents_by_type:
            fertile = self.agents_by_type[Prey].select(lambda a: a.energy > 8)
            fertile.do("reproduce")

        # Remove dead agents (energy depleted)
        dead = self.agents.select(lambda a: a.energy <= 0)
        for agent in dead:
            agent.remove()


eco = EcosystemModel(50, 5)
eco.run_for(10)
n_prey = len(eco.agents_by_type.get(Prey, []))
n_pred = len(eco.agents_by_type.get(Predator, []))
print(f"After 10 steps: {n_prey} prey, {n_pred} predators, {len(eco.agents)} total")

eco.run_for(20)
n_prey = len(eco.agents_by_type.get(Prey, []))
n_pred = len(eco.agents_by_type.get(Predator, []))
print(f"After 20 steps: {n_prey} prey, {n_pred} predators, {len(eco.agents)} total")