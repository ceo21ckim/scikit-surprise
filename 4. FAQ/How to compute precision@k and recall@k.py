from collections import defaultdict

from surprise import AlgoBase 
from surprise import SVD
import numpy as np
from surprise.dataset import Dataset
from surprise.model_selection import KFold

def precision_recall_at_k(predictions, k = 10, threshold = 3.5):
    
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))
    
    precision = dict()
    recall = dict()
    
    for uid, user_ratings in user_est_true.items(): # user_ratings : (est, true_r)
        
        # sort user ratings by estimated value
        user_ratings.sort(key = lambda x: x[0], reverse=True)
        
        # Number of relevant items 
        n_rel = sum((true_r) >= threshold for (_, true_r) in user_ratings) # all_purchase
        
        # Number of recommended items in top k
        n_rec_k  = sum((est >= threshold) for (est, _) in user_ratings[:k])
        
        # Number of relevant and recommended items in top k 
        n_rel_and_rec_k = sum((true_r >= threshold) and (est >= threshold)for (est, true_r) in user_ratings[:k])
        
        precision[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        
        recall[uid] = n_rec_k / n_rel if n_rel != 0 else 0
        
    return precision, recall
           

data = Dataset.load_builtin('ml-100k')
kf = KFold(n_splits = 5)
algo = SVD()

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k = 5, threshold=4)
    
    print( sum(prec for prec in precisions.values()) / len(precisions))
    print( sum(rec for rec in recalls.values()) / len(recalls))
