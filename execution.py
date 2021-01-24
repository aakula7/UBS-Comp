from data_collect_utils import *
from exctraction_utils import *
from clustering_utils import *
from eda_utils import *
from arima_model_utils import *
from lstm_model_utils import *

zipCodes = input('Please Enter The Zipcodes You Would LIke To Analyze:').split()
resName = input('Please Enter The Restaurant You Would Like To Analyze: ')
user = 'USERNAME'
word = 'PASSWORD'

resAttrDF, resReviewDF, dohmhDF = extract(zipCodes = zipCodes, resName = resName, host = 'localhost', user = user, word = word, attrLoadDir = None, reviewLoadDir = None, dohmhLoadDir = None, plot = True, latitude = 40.7393, longitude = -74.0020)
clustDF = dataClust(resAttrDF = resAttrDF, infCol = 'Dollars', resName = resName)
revenueDF = generateRevenue(resRev = resReviewDF, clustDF = clustDF, dohDF = dohmhDF)

plotHist(revDF = revenueDF, plotMA = True, plotInterval = True, window = 30, figsize = (15, 5))
findTrend(revDF = revenueDF, degree = 1, plot = True, figsize = (15, 5))
findOut(revDF = revenueDF, perc = 0.01, figsize = (15, 5))

arimaModeling(revDF = revenueDF, resName = resName, resample = 'MS', model = 'additive', s = 12, max_p = 3, max_d = 3, max_q = 3, max_P = 3, max_D = 3, max_Q = 3, seasonal = True, stationary = False, figsize = (12, 5))
anomallyDet(revDF = revenueDF, resName = resName, valid = None, test = 0.25, plot = True, figsize = (15, 5), n_steps = 30, units = 64, dropout = 0.2, optimizer = 'adam', metrics = 'accuracy', batch_size = 32, loss = 'mae', epochs = 100)
multiVariate(revDF = revenueDF, resName = resName, valid = None, test = 0.25, plot = True, figsize = (15, 5), look_back = 3, dropout = 0.2, lstmAct = 'softmax', denseAct = 'sigmoid', optimizer = 'adam', metrics = 'accuracy', batch_size = 32, loss = 'mae', epochs = 100)