# The basics
from surprise import AlgoBase 
from surprise import Dataset 
from surprise.model_selection import cross_validate
import numpy as np 

# AlgoBase => nn.Module
class MyOwnAlgorithm(AlgoBase):
    def __init__(self):
        
        # Always call base method befor doing anything.
        AlgoBase.__init__(self)
    
    def estimate(self, u, i):
        return 3
    
data = Dataset.load_builtin('ml-100k')
algo = MyOwnAlgorithm()

cross_validate(algo, data, verbose = True)


class MyModel(AlgoBase):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def estimate(self, u, i):
        
        details = {'info1' : 'That was', 
                   'info2' : 'easy stuff'}
        return 3, details
    
algo = MyModel()

cross_validate(algo, data, verbose = True)


# fit method 
# The fit method is called e.g. by the cross_validate function at each fold of a cross-validation process
# Before doing anything, you should call the base class fit() method.

class myModel(AlgoBase):
    def __init__(self):
        super(myModel, self).__init__()
        
    def fit(self, trainset):
        
        # Here again : call base method befor doing anything.
        AlgoBase.fit(self, trainset)
        
        # Compute the average rating. We might as well use the trainset.global_mean attribute ;
        self.the_mean = np.mean([r for (_, _, r) in self.trainset.all_ratings()])
        
        return self 
    
    def estimate(self, u, i):
        return self.the_mean

