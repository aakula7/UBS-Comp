import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from kmodes.kprototypes import KPrototypes
from keras.layers import Embedding
import os
import lightgbm as lgb
from matplotlib import pyplot as plt
import shap

#########################################################################################################
## --------------------------------------  DATA CLUSTERING  ------------------------------------------ ##
#########################################################################################################
def dataClust(resAttrDF, infCol = 'Dollars', resName = None):
    """
    CLUSTERING YELP RESTAURANT ATTRIBUTE DATA ACCORDING TO COLUMN PROVIDED
    
    :Inputs
        :param resAttrDF: Restaurant attribute data for clustering
        :param infCol: Column to use for number of clusters, DEFAULT: 'Dollars'
        :param resName: Restaurant name that the user is trying to analyze

    :Return
        :return k_clust: Clustered data on restaurant attributes
    """
    
    if resName is None:
        raise Exception('**** RESTAURANT NAME WAS NOT PROVIDED ****')
    
    ## COPY AND PREPROCESS RESTAURANT ATTRIBUTE DATA
    print(f'\n**** PREPROCESSING AND CLUSTERING DATA ACCORDING TO...{infCol.upper()} COLUMN ****')

    k_clust = resAttrDF.copy()
    k_clust = k_clust.reset_index(drop = True)
    
    labelEncoder = LabelEncoder()
    k_clust['Name'] = labelEncoder.fit_transform(k_clust['Name'])
    for col in k_clust.columns:
        if k_clust[col].dtypes == 'object':
            k_clust[col] = pd.to_numeric(k_clust[col])

    kprot_data = k_clust.copy()
    for c in k_clust.select_dtypes(exclude='object').columns:
        pt = PowerTransformer()
        kprot_data[c] =  pt.fit_transform(np.array(kprot_data[c]).reshape(-1, 1))

    categorical_columns = [0] ## MAKE SURE TO SPECIFY CURRECT INDICES

    ## ACTUAL CLUSTERING
    if infCol != 'Dollars':
        kproto = KPrototypes(n_clusters= len(k_clust[infCol].unique()), init='Cao', n_jobs = 4)
        clusters = kproto.fit_predict(kprot_data, categorical=categorical_columns)
    else:
        kproto = KPrototypes(n_clusters= len(k_clust['Dollars'].unique()), init='Cao', n_jobs = 4)
        clusters = kproto.fit_predict(kprot_data, categorical=categorical_columns)       

    ## PRINT COUNT OF EACH CLUSTER GROUP
    print('The count for each cluster group is printed below')
    pd.Series(clusters).value_counts()
    
    ## EVALUATE CLUSTER ACCURACY WITH LGBMCLASSIFIER
    clf_kp = lgb.LGBMClassifier(colsample_by_tree=0.8, random_state=1)
    cv_scores_kp = cross_val_score(clf_kp, k_clust, clusters, scoring='f1_weighted')
    print(f'CV F1 score for K-Prototypes clusters is {np.mean(cv_scores_kp)}')

    ## PLOT INFLUENTIAL COLOUMNS
    clf_kp.fit(k_clust, clusters)
    explainer_kp = shap.TreeExplainer(clf_kp)
    shap_values_kp = explainer_kp.shap_values(k_clust)
    shap.summary_plot(shap_values_kp, k_clust, plot_type="bar", plot_size=(15, 10))

    ## ADD CLUSTERS TO ORIGINAL DATAFRAME AND INVERSE LABEL ENCODE RESTAURANT NAMES
    k_clust['Cluster'] = clusters
    k_clust['Name'] = labelEncoder.inverse_transform(k_clust['Name'])

    ## FILTER RESTAURNAT CLUSTER OF CHOICE
    clusterVal = clusters[list(k_clust['Name']).index(resName)]
    k_clust = k_clust[k_clust['Cluster'] == clusterVal]
    k_clust = k_clust.reset_index(drop = True)
    k_clust = k_clust[['Name', 'ZipCode', 'Dollars', 'Photos']]

    print('**** CLUSTERING COMPLETED AND SAVING CLUSTER DATAFRAME LOCALLY ****\n')
    resFileName = resName.replace(' ', '_')
    fileName = f'{resFileName.upper()}_CLUSTER_DATA.csv'
    k_clust.to_csv(fileName)

    return k_clust