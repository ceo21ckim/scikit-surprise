# scikit-surprise
# Automatic cross-validation 

import surprise
from surprise import SVD
from surprise import Dataset 
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# we'll use the famus SVD algorithm
svd = SVD()

# Run 5-fold cross-validation and print results 
cross_validate(svd, data, measures = ['RMSE', 'MAE'], cv = 5, verbose = True)



# Train-test split and the fit() method 

from surprise import (
    SVD, 
    Dataset, 
    accuracy
)
from surprise.model_selection import train_test_split

train_set, test_set = train_test_split(data, test_size = 0.25, random_state=42)


svd = SVD()

svd.fit(train_set)

predictions = svd.test(test_set)

# compute RMSE 
accuracy.rmse(predictions)

# Train on a whole trainset and the predict() method 
from surprise import (
    KNNBasic, 
    Dataset 
)

trainset = data.build_full_trainset()

knn = KNNBasic()
knn.fit(trainset)

user_id = str(196)
item_id = str(302)

# rating = 4 ! 
pred = knn.predict(user_id, item_id, r_ui = 4, verbose = True)

# Use a custom dataset

from surprise import (
    BaselineOnly, 
    Dataset, 
    Reader
)
from surprise.model_selection import cross_validate
import os 
file_path = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/u.data')

reader = Reader(line_format = 'user item rating timestamp', sep = '\t')

data = Dataset.load_from_file(file_path, reader = reader)

# We can now use this dataset as we please, e.g. calling cross_validate
cross_validate(BaselineOnly(), data, verbose = True)


import pandas as pd
from surprise import (
    NormalPredictor, 
    Dataset, 
    Reader
)
from sklearn.model_selection import cross_validate

# Create of the dataframe, Column names are irrelevant.
ratings_dict = {'itemID' : [1, 1, 1, 2, 2], 
                'userID' : [9, 32, 2, 45, 'user_foo'],
                'rating' : [3, 2, 4, 3, 1]
                }


df = pd.DataFrame(ratings_dict)

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale = (1, 5)) # minimum rating = 1, maximum rating = 5

data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)

np = NormalPredictor()

cross_validate(np, data, cv = 5 )


# Use Cross-validation iterators

from surprise import (
    SVD, 
    Dataset, 
    accuracy
)
from surprise.model_selection import KFold

#  Load the movielens-100k dataset
data = Dataset.load_builtin('ml-100k')

# define a cross-validation iterator 
kf = KFold(n_splits = 3)

svd = SVD()

for trainset, testset in kf.split(data):
    
    _ = svd.fit(trainset)
    predictions = svd.test(testset)
    
    # Compute and print Root Mean Squared Error 
    _ = accuracy.rmse(predictions, verbose = True)
    
    
from surprise.model_selection import PredefinedKFold

# path to dataset folder
files_dir = os.path.expanduser('~/.surprise_data/ml-100k/ml-100k/')

# This time, we'll use the bult-in reader.
reader = Reader('ml-100k')

# folds_files is a list of tuples containing file paths:
# [(u1.base, u1.test), (u2.base, u2.test), ... , (u5.base, u5.test)]
train_file = files_dir + 'u%d.base'
test_file = files_dir + 'u%d.test'

folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

data = Dataset.load_from_folds(folds_files, reader = reader)
pkf = PredefinedKFold()

svd = SVD()

for trainset, testset in pkf.split(data):
    
    _ = svd.fit(trainset)
    predictions = svd.test(testset)
    
    _ = accuracy.rmse(predictions, verbose = True)
    


# Tune algorithm parameters with GridSearchCV

from surprise.model_selection import GridSearchCV

data = Dataset.load_builtin('ml-100k')

param_grid = {'n_epochs' : [5, 10], 'lr_all' : [0.002, 0.005],
              'reg_all' : [0.4, 0.6]}

gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv = 3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])

algo = gs.best_estimator['rmse']

algo.fit(data.build_full_trainset())


results_df = pd.DataFrame.from_dict(gs.cv_results)

results_df

pd.DataFrame(gs.cv_results)