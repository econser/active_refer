import numpy as np
import agent as a

class AgentPool(object):
    def __init__(self, min_agent_count, objects):
        """ init
        min_agent_count: mininum number of agents for the pool
        objects: a list of the scenario's objects
        """
        self._min_agent_count = min_agent_count
        self._objects = objects
        self._pool = []

        while self.get_agent_count() < self._min_agent_count:
            self.add_agent()

    def add_agent(self):
        entity_id = np.random.randint(len(self._objects))
        if np.random.randint(100) < 50:
            self._pool.append(a.DetectorAgent(entity_id))
        else:
            self._pool.append(a.SituationAgent(entity_id))

    def get_agent(self):
        self.add_agent()
        return self._pool.pop(0)

    def get_agent_count(self):
        return len(self._pool)
