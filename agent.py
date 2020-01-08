import numpy as np

class Agent(object):
    """ Agents are responsible for generating bboxes and determining their internal support strength
    """
    def __init__(self, object_id):
        self.object_id = object_id
        self.internal_support = None
        self.bbox = None

    def get_internal_support(self):
        if self.bbox is None:
            self.bbox = self.sample_unit_bbox()
            
        if self.internal_support is None:
            self.internal_support = np.random.beta(1,3)
        return self.internal_support

    def sample_unit_bbox(self):
        # returns a unit bbox, rescale to the image size
        if self.bbox is None:
            self.bbox = np.random.beta(1,3,(1,4))
        return self.bbox



class DetectorAgent(Agent):
    """ Use this method to sample a bbox independent of the situation, based only on CNN confidence scores
    """
    pass



class SituationAgent(Agent):
    """ Use this method to sample a bbox given the x,y,aspect,area of the other situation objects
    """
    pass



class PriorAgent(Agent):
    """ Use this method when no object is alread active
    """
    pass
