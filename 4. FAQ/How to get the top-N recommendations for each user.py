# Here is an example where we retrieve the top-10 items with highest rating prediction for each user in the MovieLens-100k dataset.
# We first train an SVD algorithm on the whole dataset, and then predict all the ratings for the pairs (user, item) that are not in the training set.
# We then retrieve the top-10 prediction for each user.

from collections import defaultdict
# defaultdict는 dictionary 와 거의 비슷하지만, key값에 없는 값을 입력할 경우 미리 지정해놓은 default값을 반환함.

from surprise import SVD
from surprise import Dataset 


# 모델이 출력된 후 나온 결과값을 prediction에 넣어주면 된다. 
'''
predictions output : uid, iid, r_ui(True_r), est(pred_r), details = {'was_imposstible': False}
'''
def get_top_n(predictions, n = 10):
    '''Return the top-N recommendation for each user from a set of predictions.
    
    
    Args: 
        predictions(list of Prediction objects): The list of prediction, as returned by the test method of an algorithm.
        n(int): The number of recommendation to ouput for each user. Default is 10.
        
        
    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    '''
    
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
        
    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key = lambda x: x[1], reverse = True)
        top_n[uid] = user_ratings[:n]
        
    return top_n

# First train an SVD algorithm on the movielens dataset.
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Then predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)



top_n = get_top_n(predictions, n = 10)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])

