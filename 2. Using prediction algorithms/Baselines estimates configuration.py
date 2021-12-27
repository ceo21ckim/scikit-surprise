from surprise import KNNBasic
from surprise.prediction_algorithms.baseline_only import BaselineOnly

knn = KNNBasic()

# ALS
# reg_i : The regularization parameter for items. Default is 10
# reg_u : The regularization parameter for users. Default is 15
# n_epochs : The number of iteration of the ALS procedure. Default is 10 

bsl_options = {'method': 'als', 
               'n_epochs' : 5, 
               'reg_i' : 5, 
               'reg_u' : 12}

als = BaselineOnly(bsl_options = bsl_options )

# SGD
# 'reg' : The regularization parameter of the cost function that is optimized. Default is 0.02
# 'learning_rate' : The learning rate of SGD, corresponding to gamma. Default is 0.005
# 'n_epochs' : The number of iteration of the SGD. Default if 20.

bsl_options = {'method' : 'sgd', 
               'learning_rate' : .00005
               }

sgd = BaselineOnly(bsl_options=bsl_options)


# Similarity measure configuration 
# 'name' : The name of the similarity to use, defined in the similarities module. Default if 'MSD'
# 'user_based' : This has a huge impact on the performance of a prediction algorithm. Default is True
# 'min_support' : The minimum number of common item (when 'user_based' is 'True')

bsl_options = {'method' : 'als', 
               'n_epochs' : 20, 
               }
sim_options = {'name' : 'pearson_baseline', 
               'user_based' : True}

knn = KNNBasic(bsl_options = bsl_options, sim_options=sim_options)

