# The trainset attribute 
# Once the base class fit() method has returned, all the info you need about the current training set (rating values, etc...) is stored in the self.trainset attribute
# This is a Trainset object that has many attributes and methods of interest for prediction.

# To illustrate its usage, let's make an algorighm that predicts an average between the mean of all ratings, the mean rating of the user and the mean rating for item:
from surprise import AlgoBase
from surprise import Dataset 
import numpy as np 

class MyModel(AlgoBase):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def fit(self, trainset):
        
        AlgoBase.fit(self, trainset)

        self.the_mean = np.mean([r for (_, _, r) in self.trainset_.all_ratings()])
        

    def estimate(self, u, i):
        sum_means = self.trainset.global_mean 
        div = 1

        if self.trainset.knows_user(u):
            sum_means += np.mean([r for (_, r) in self.trainset.ur[u]])
            div += 1
            
        if self.trainset.knows_item(i):
            sum_means += np.mean([r for (_, r) in self.trainset.ir[i]])
            div += 1
            
        return sum_means / div
    
# When the prediction is impossible 
# It's up to your algorithm to decide if it can or cannot yield a prediction. If the prediction is impossible.
# Then you can raise the "PredictImpossible" exception. You'll need to import it first

# This exception will be caught by the predict() method, and the estimation r_ui will be set according to the default_prediction() method
# which can be overridden. By default it returns the average of all ratings in the trainset.

from surprise import PredictionImpossible