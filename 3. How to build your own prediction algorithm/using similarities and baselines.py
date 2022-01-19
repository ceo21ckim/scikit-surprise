# Should your algorithm use a similarity measure or baseline estimates, you'll need to accept 'bsl_options' and 'sim_options' as parameters to the __init__ method
# and pass them along to the Base class. See how to use these parameters in the Using prediction algorithms section.
# Method compute_baselines() and compute_similarities() can be called in the fit method (or anywhere else).

from surprise import AlgoBase
from surprise import Dataset 
import numpy as np 
from surprise import PredictionImpossible

# sim -> similar, bsl -> baseline
class MyModel(AlgoBase):
    def __init__(self, sim_options = {}, bsl_options = {}):
        super(MyModel, self).__init__(self, sim_options = sim_options, bsl_options = bsl_options)
        
    def fit(self, trainset):
        
        AlgoBase.fit(self, trainset)
        
        # Compute baselines and similarities
        self.bu, self.bi = self.compute_baselines()
        self.sim = self.compute_similarities()
        
        return self 
    
    def estimate(self, u, i):
        
        if not (self.trainset.knows_user(u) and self.trainset.konws_item(i)):
            raise PredictionImpossible('User and/or item is unkown.')
        
        # Compute similarities between u and v, where v describes all other 
        # users that have also item i.
        neighbors = [(v, self.sim[u,v]) for (v, _) in self.trainset.ir[i]]
        
        # Sort these neighbors by similarity 
        neighbors = sorted(neighbors, key = lambda x: x[1], reverse = True)
        
        print('The 3 nearest neighbors of user', str(u), 'are:')
        for v, sim_uv in neighbors[:3]:
            print('user {0:} with sim {1:1.2f}'.format(v, sim_uv))

