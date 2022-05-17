import habitat_sim
from agents import Sem_Exp_Env_Agent
from numpy import bool_
import habitat
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions

class Sem_Agent(habitat.Agent):

    def reset(self):
        pass
    
    def act(self, observations: Observations):
        

