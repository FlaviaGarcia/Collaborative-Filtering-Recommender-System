#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: flaviagv

Script for training, evaluating and testing a Collaborative Filtering (CF)
recommender using as dataset Million Song Dataset. The algorithm used form
this recommender development had been the Alternating Least Squares(ALS), 
fixing the following hyperparameters:
    - regularization hyperparameter -> 0.01
    - number of latent factors      -> 130
    - number of iterations          -> 30


"""

import pandas as pd
import scipy.sparse as sparse
import numpy as np
import implicit
from six.moves import cPickle as pickle
import random
from sklearn.metrics import average_precision_score as AP



def LoadPickleFile(pickle_file = "dataset.pickle"):
    """Load a pickle filed
    Args:
        pickle_file: Pickle file path
    Returns:
        Loaded pickle file
    
    """
    f        = open(pickle_file, "rb")
    Datasets = pickle.load(f)
    
    return Datasets



def readInputs(path_dataset):
    """Load input triplets table 
    Args:
        path_dataset: path input dataset
    Returns:
        pandas with the loaded dataframe that has the following columns:
            - user
            - song
            - play_count
    
    """
    print("-> Reading input data ...")
    EchoNestDB = pd.read_table(path_dataset, header = None)
    EchoNestDB.columns =["user", "song", "play_count"]
    return EchoNestDB



def get_and_split_Subset(EchoNestDB, NUM_SONGS, NUM_USERS):
    """Get a subset of the whole dataset and it splits it in train (80%), 
    validation (10%) and test (10%).
    Args:
        EchoNestDB: pandas input dataset
        NUM_SONGS:  number of songs of the subset
        NUM_USERS:  number of users of the subset
    Returns:
        trainSparse
        ValSparse 
        testSparse
        users:       row names of the previous sparse matrices (users ids)
        songs:       column names of the previous sparse matrices (songs ids)
        song_users_altered: users that have information in the validation and test sets
    
    """
    print("-> Getting the subset and fetching its data separation ...")
    # Getting the subset with NUM_SONGS songs and NUM_USERS users
    rep_songs = pd.DataFrame({'count' : EchoNestDB.groupby("song").size()}).reset_index().sort_values("count", ascending=False)
    songs_selected = rep_songs["song"][:NUM_SONGS]
    dfFilter_tmp = EchoNestDB[:][EchoNestDB.song.isin(songs_selected)]
    rep_users = pd.DataFrame({'count' : dfFilter_tmp.groupby("user").size()}).reset_index().sort_values("count", ascending=False)
    users_20000 = rep_users["user"][:NUM_USERS]
    dfFilter = dfFilter_tmp[:][dfFilter_tmp.user.isin(users_20000)]
        
    
    matrix_size = NUM_SONGS * NUM_USERS
    sparsity = 100 * (1 - (len(dfFilter)/matrix_size))
    print("Sparsity whole dataset matrix %f"%sparsity) 
    
    # We random the resulting subset and it is splitted in train (80%), validation (10%) and test(10%)
    dfFilter_rand = dfFilter.sample(frac = 1, random_state = 200) 
    limit_set = dfFilter_rand.shape[0]*8//10  # 80%
    limit_set_val = dfFilter_rand.shape[0]*1//10 # 10%
    
    
    ## Train matrix sparsity
    matrix_size = NUM_SONGS * NUM_USERS
    sparsity = 100*(1 - (limit_set/matrix_size))
    print("Sparsity of train matrix %f"%sparsity) 
    

    # DataFrames of each set 

    dfPlayCounts_train = dfFilter_rand[:(limit_set + 1)] # (3.158.642, 3)

    trainDf       = dfPlayCounts_train.pivot(index = 'user', columns ='song', values = 'play_count').fillna(0)
    validationDf  = dfFilter_rand[:(limit_set + 2 + limit_set_val)].pivot(index = 'user', columns ='song', values = 'play_count').fillna(0)
    testDf        = dfFilter_rand.pivot(index = 'user', columns ='song', values = 'play_count').fillna(0)
    validationDf[trainDf != 0] = 0     
    testDf[trainDf != 0] = 0 
    testDf[validationDf != 0] = 0 
    

    #### Sum play counts of songs that actually are the same and delete that extra song columns    
    dictCleaning = LoadPickleFile("../00-Data_in/dictDuplicted.pickle")
    
    # Train
    columnsDeleted = []
    i = 0 
    notKey = []
    for key in dictCleaning.keys():
        # print(str(i))
            
        listDeleted = dictCleaning[key] # Lista de los songs ids que eliminaremos
        
        songsRep = np.unique(np.asarray([key] + listDeleted)).tolist()
        
        listenedSongs = list(set(songsRep).intersection(set(trainDf.columns)))
        
        
        if len(listenedSongs) == 0:
            notKey.append(key)
            # print("Not any song of repeated in the subset")
        else:
            #break  # la primera vez que entre aqui
            newKey      = listenedSongs[0]
            summedSongs = trainDf[listenedSongs].sum(axis = 1) ## tiene que ser 10.000 de len
            trainDf[newKey] = summedSongs
            
            columnsDeleted = columnsDeleted + listenedSongs[1:]
       
        i += 1    
        
    print("The columns deleted because of being the same song of other already in the dataset" +
          " are: %i"%len(columnsDeleted))
            
    trainDf_new = trainDf.drop(columnsDeleted, axis = 1)    
    
    # Validation
    columnsDeleted = []
    i = 0 
    notKey = []
    for key in dictCleaning.keys():
        # print(str(i))
            
        listDeleted = dictCleaning[key] # Lista de los songs ids que eliminaremos
        
        songsRep = np.unique(np.asarray([key] + listDeleted)).tolist()
        
        listenedSongs = list(set(songsRep).intersection(set(validationDf.columns)))
        
        #listenedKey = EchoNestDB_new[EchoNestDB_new.song == key] ## Nos tenemos que recorrer sus users
        
        if len(listenedSongs) != 0:
            #break  # la primera vez que entre aqui
            newKey      = listenedSongs[0]
            summedSongs = trainDf[listenedSongs].sum(axis = 1) ## tiene que ser 10.000 de len
            validationDf[newKey] = summedSongs
            
            columnsDeleted = columnsDeleted + listenedSongs[1:]
       
        i += 1    
                    
    validationDf_new = validationDf.drop(columnsDeleted, axis = 1)    
    
    
    #Test
    columnsDeleted = []
    i = 0 
    notKey = []
    for key in dictCleaning.keys():
        # print(str(i))
            
        listDeleted = dictCleaning[key] # Lista de los songs ids que eliminaremos
        
        songsRep = np.unique(np.asarray([key] + listDeleted)).tolist()
        
        listenedSongs = list(set(songsRep).intersection(set(testDf.columns)))
        
        #listenedKey = EchoNestDB_new[EchoNestDB_new.song == key] ## Nos tenemos que recorrer sus users
        
        if len(listenedSongs) != 0:
            #break  # la primera vez que entre aqui
            newKey      = listenedSongs[0]
            summedSongs = testDf[listenedSongs].sum(axis = 1) ## tiene que ser 10.000 de len
            testDf[newKey] = summedSongs
            
            columnsDeleted = columnsDeleted + listenedSongs[1:]
       
        i += 1    
                    
    testDf_new = testDf.drop(columnsDeleted, axis = 1)    
    
    testDf_new[testDf_new != 0] = 1
    validationDf_new[validationDf_new != 0] = 1
    
    users = list(trainDf_new.index) 
    songs = list(trainDf_new.columns)
    
    bool_array = (validationDf_new != 0).any(axis = 1).values 
    song_users_altered = np.where(bool_array == True)[0].tolist()
    # Each set matrix is converted to sparse
    trainSparse = sparse.csc_matrix(trainDf_new)
    ValSparse  = sparse.csc_matrix(validationDf_new)
    testSparse  = sparse.csc_matrix(testDf_new)
     
    return (trainSparse, ValSparse, testSparse, users, songs, song_users_altered)



def als(trainSparse):
    """Training of the ALS algorithm
    Args:
        trainSparse: train sparse matrix

    Returns:
        user_vecs_arr: user matrix (users x latent_factors)
        item_vecs_arr: item matrix (items x latent_factors)
    
    """
    print("-> Training ALS algorithm ...")
    k = 130 
    user_vecs_arr, item_vecs_arr = implicit.alternating_least_squares(trainSparse, 
                                                                  factors = k, 
                                                                  regularization = 0.01, 
                                                                  iterations = 30)
    return(user_vecs_arr, item_vecs_arr)



def getRankingPos_test(user, training_set, predictions_list, test_set, ran = False):
    """ Get ranking position of the songs in the test_set
    Args:
        user: user wanted to get its songs ranking position
        training_set: training set sparse matrix
        predictions_list: list with users matrix(position 0) and the items matrix (position 1)
        test_set: test set sparse matrix
        ran = False: True if the random recommender wants to be used
    Returns: 
        ranking_testSongs: list with the number of the positionin the ranking of 
        the songs listened in test_set
        
    """
    item_vecs = predictions_list[1]
    
    #for user in range(training_set.shape[0]):
    # for user in altered_users: # Iterate through each user that had an item altered
    # productos de este user
    training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        
    
    zero_inds = np.where(training_row == 0) # Posiciones las cuales mirare sus predcciones

    # Get the predicted values based on our user/item vectors
    # fila de la matriz con los latentes de ese user
    if ran == False:
        user_vec = predictions_list[0][user,:]
            
        # prediccion de productos de ese user
        predictions = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
    else:
        predictions= [random.random() for test in range(training_set.shape[1])]
        
    predictions_df = pd.DataFrame(predictions)
    # Ordenamos las predicciones de forma descendente y le asignamos un numero 
        # con la posicion en nuestro ranking 
    predictions_df = predictions_df.sort_values(by = 0, ascending = False)        
    predictions_df.columns = ["Predictions"]
    predictions_df["num_ranking"] = range(1, predictions_df.shape[0] + 1)
    
            
    test    = test_set[user,:].toarray()[0,zero_inds].reshape(-1) 
    test_df = pd.DataFrame(test)
    heardSongs_test = test_df[test != 0]
    ranking_testSongs = predictions_df.loc[heardSongs_test.index, "num_ranking"]
            # Muy abajo en el ranking con el ejemplo del user 0  
        
        
    return ranking_testSongs  ## Devuelve las canciones del test que numero del ranking se le ha sido asignadas



def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.



def mAP_nDCG_k(reconstTrainMatrix, trainSparse, altered_users, testSparse, k=10, ran = False):
    """Calculate the evaluation metrics mAP and nDCG of the top-k recommended list
    Args:
        reconstTrainMatrix: Matrix obtained by doing the dot product of the 
                            users matrix and the items matrix (ALS results)
        trainSparse:        Train sparse matrix
        altered_users:      User indices that have played songs in the test matrix 
        testSparse:         Test sparse matrix
        k:                  Number of songs in the recommended list
        ran = False:        True if the random recommender wants to be evaluated
    Returns:
        mAP and nDCG of the performance of the whole recommender system
    """
    sum_AP = 0
    sum_nDCG = 0
    # Hacerlo para cada usuario    
    for user in altered_users:
        # print(str(user))
        # if(sum(testSparse.toarray()[user,:])>=k):
        if ran == False:
            y_score = reconstTrainMatrix[user,:].tolist()
        else: 
            y_score = [random.random() for test in range(trainSparse.shape[1])]
        y_true  = testSparse.toarray()[user,:]
        df      = pd.DataFrame({"y_true": y_true.tolist(), "y_score": y_score})
        df_orderedPred = df.sort_values(by = "y_score", ascending=False)
        
        
        
                
        y_true_k  = df_orderedPred["y_true"].values[:k]
        y_score_k = df_orderedPred["y_score"].values[:k]
        
        av_prec = AP(y_true_k, y_score_k)
        if str(av_prec) == "nan":
            av_prec = 0
        sum_AP += av_prec
        
        dcg = dcg_at_k(y_true_k, k)               
        dcg_max = dcg_at_k(np.ones(k), k)   # todos los elementos de la lista son 1
        sum_nDCG += dcg/dcg_max
    
           
    mAP = sum_AP/len(altered_users)
    
    nDCG = sum_nDCG/len(altered_users)

    
    return mAP, nDCG



def precision_recall_F1_at_k(training_set, altered_users, predictions_list, test_set, k=10 ,ran = False):
    """Calculate the evaluation metrics: recall, precision and f1 score of the top-k recommended list
    Args:
        training_set:     Train sparse matrix
        altered_users:    User indices that have played songs in the test matrix 
        predictions_list: List with users matrix(position 0) and the items matrix (position 1)
        test_set:         Test sparse matrix
        k:                Number of songs in the recommended list
        ran = False:      True if the random recommender wants to be evaluated
    Returns:
        mAP and nDCG of the performance of the whole recommender system
    """
    

    # First map the predictions to each user.

    precisions = []
    recalls = []
    f1 = []
    for user in range(training_set.shape[0]):#altered_users: # el index de la matriz de ese user
        # print(str(user))
        train = False
        # if(sum(test_set.toarray()[user,:])>=k):
        rankingTestSongs = getRankingPos_test(user, training_set, predictions_list, test_set, train, ran)
           
        size_HitSet = sum(rankingTestSongs <= k )
        size_testSet = len(rankingTestSongs)
        recall = size_HitSet/size_testSet
        precision = size_HitSet/k
        F1 = 2 * precision * recall / (recall + precision)
        if str(F1) == "nan":
            F1 = 0
        recalls.append(recall)
        precisions.append(precision)
        f1.append(F1)
    # pasar las listas a arrays, y hacer su media 
    return np.mean(np.asarray(precisions)), np.mean(np.asarray(recalls)), np.mean(np.asarray(f1))




if __name__ == "__main__":
    
    inputMatrix  = readInputs("../00-Data_in/train_triplets.txt")
    trainSparse, ValSparse, testSparse, users, songs, song_users_altered =  \
    get_and_split_Subset(inputMatrix, NUM_SONGS = 5000, NUM_USERS = 10000)

    user_vecs_arr, item_vecs_arr = als(trainSparse)

    reconstTrainMatrix = np.dot(user_vecs_arr, item_vecs_arr.transpose())

    print ("---- VALIDATION RESULTS ----")
    # TOP 10 
    precision, recall, f1 = precision_recall_F1_at_k(trainSparse, song_users_altered, [sparse.csr_matrix(user_vecs_arr), sparse.csr_matrix(item_vecs_arr.T)], ValSparse)
    print("precision@10: %f, recall@10: %f, F1@10: %f"%(precision,recall, f1)) 
    mAP, nDCG = mAP_nDCG_k(reconstTrainMatrix, trainSparse, song_users_altered, ValSparse)
    print("mAP@10: %f, nDCG@10: %f"%(mAP,nDCG)) 
    # TOP 20
    precision_20, recall_20, f1_20 = precision_recall_F1_at_k(trainSparse, song_users_altered, [sparse.csr_matrix(user_vecs_arr), sparse.csr_matrix(item_vecs_arr.T)], ValSparse, 20)     
    print("precision@20: %f, recall@20: %f, F1@20: %f"%(precision_20, recall_20, f1_20)) 
    mAP_20, nDCG_20 = mAP_nDCG_k(reconstTrainMatrix, trainSparse, song_users_altered, ValSparse, 20)
    print("mAP@20: %f, nDCG@20: %f"%(mAP_20,nDCG_20))     
    # TOP 30 
    precision_30, recall_30, f1_30 = precision_recall_F1_at_k(trainSparse, song_users_altered, [sparse.csr_matrix(user_vecs_arr), sparse.csr_matrix(item_vecs_arr.T)], ValSparse, 30)
    print("precision@30: %f, recall@30: %f, F1@30: %f"%(precision_30, recall_30, f1_30)) 
    mAP_30, nDCG_30 = mAP_nDCG_k(reconstTrainMatrix, trainSparse, song_users_altered, ValSparse, 30)
    print("mAP@30: %f, nDCG@30: %f"%(mAP_30,nDCG_30)) 


    print ("---- TEST RESULTS ----")
    # TOP 10 
    precision, recall, f1 = precision_recall_F1_at_k(trainSparse, song_users_altered, [sparse.csr_matrix(user_vecs_arr), sparse.csr_matrix(item_vecs_arr.T)], testSparse)
    print("precision@10: %f, recall@10: %f, F1@10: %f"%(precision, recall, f1)) 
    mAP, nDCG = mAP_nDCG_k(reconstTrainMatrix, trainSparse, song_users_altered, testSparse)
    print("mAP@10: %f, nDCG@10: %f"%(mAP, nDCG)) 
    # TOP 20 
    precision_20, recall_20, f1_20 = precision_recall_F1_at_k(trainSparse, song_users_altered, [sparse.csr_matrix(user_vecs_arr), sparse.csr_matrix(item_vecs_arr.T)], testSparse, 20)     
    print("precision@20: %f, recall@20: %f, F1@20: %f"%(precision_20, recall_20, f1_20)) 
    mAP_20, nDCG_20 = mAP_nDCG_k(reconstTrainMatrix, trainSparse, song_users_altered, testSparse, 20)
    print("mAP@20: %f, nDCG@20: %f"%(mAP_20,nDCG_20))     
    # TOP 30 
    precision_30, recall_30, f1_30 = precision_recall_F1_at_k(trainSparse, song_users_altered, [sparse.csr_matrix(user_vecs_arr), sparse.csr_matrix(item_vecs_arr.T)], testSparse, 30)
    print("precision@30: %f, recall@30: %f, F1@30: %f"%(precision_30,recall_30, f1_30)) 
    mAP_30, nDCG_30 = mAP_nDCG_k(reconstTrainMatrix, trainSparse, song_users_altered, testSparse, 30)
    print("mAP@30: %f, nDCG@30: %f"%(mAP_30,nDCG_30)) 











