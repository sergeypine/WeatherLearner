import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA  

pd.set_option('display.max_rows', 250)

#=====================================================================
def exploreModel(featuresetFile, n):

	predictedVariable = parseVarnameFname(featuresetFile)
	predictionHoursAhead = parsePredictionIntervalFname(featuresetFile)
	lookBackHours = parseLookbackWindowFname(featuresetFile)
	
	featureset = pd.read_csv(featuresetFile, parse_dates=['prediction_for_ts'])
	featureset = featureset.dropna()

	print("Loaded file for VAR = {}, predictionInterval = {}h, lookback = {}h".format(predictedVariable, predictionHoursAhead, lookBackHours))
	#columns_to_drop = []
	#for col in featureset.columns:
	#	if 'intensity' not in col and 'DewPoint' and 'Wind' not in col and col != 'Y' and col != 'prediction_for_ts':
	#		columns_to_drop.append(col)
	#featureset = featureset.drop(columns = columns_to_drop)
	#print("Dropped {} columns, {} remaining".format(len(columns_to_drop), len(featureset.columns)))
	
	#coll = featureset.corr()[['Y']].sort_values('Y')
	#for i, row in coll.iterrows():
	#	if abs(row['Y']) > 0.2:
	#		print(row)
	#return

	#  normalize data
	X = featureset.drop(columns = ['Y', 'prediction_for_ts'])
	scaler = MinMaxScaler() 
	X = scaler.fit_transform(X)

	# Use PCA to shrink feature count
	feature_reduced_X = performPCAAnalysis(X, n)
	feature_reduced_X = pd.DataFrame(data = feature_reduced_X, columns = ["col{}".format(i) for i in range(0, n)])


	#(3) Train/Test Split
	X_train, X_test = piecewiseSplit(feature_reduced_X, 15, 3, 3)
	print("Piecewise Split: trainset len = {}, testset_len = {}".format(len(X_train), len(X_test)))
	Y_train, Y_test = piecewiseSplit(featureset[['Y']], 15, 3, 3)

	if "is_" in predictedVariable: # Classification
		regressor = MLPClassifier(hidden_layer_sizes = [int(len(X_train.columns) * 0.95), int(len(X_train.columns) * 0.9), int(len(X_train.columns) * .7), int(len(X_train.columns) * .4)], early_stopping = True)
		#regressor = xgb.XGBClassifier(eta = 0.05, n_estimators = 4000, max_depth = 10, eval_metric = 'error')
		regressor.fit(X_train, Y_train.values.ravel())

		Y_predict = regressor.predict(X_test)
		score = f1_score(Y_test, Y_predict)
		mcc = matthews_corrcoef(Y_test, Y_predict)
		print("TEST: F1 Score for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, score, predictionHoursAhead, lookBackHours))
		print("TEST: MCC Score for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, mcc, predictionHoursAhead, lookBackHours))
		print(confusion_matrix(Y_test, Y_predict))

		Y_predict = regressor.predict(X_train)
		score = f1_score(Y_train, Y_predict)
		mcc = matthews_corrcoef(Y_train, Y_predict)	
		print("TRAIN: F1 Score for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, score, predictionHoursAhead, lookBackHours))
		print("TRAIN: MCC Score for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, mcc, predictionHoursAhead, lookBackHours))
		print(confusion_matrix(Y_train, Y_predict))

	else: # Regression
		regressor = MLPRegressor(hidden_layer_sizes = [int(len(X_train.columns) * 0.95), int(len(X_train.columns) * 0.9), int(len(X_train.columns) * .7), int(len(X_train.columns) * .4)], early_stopping = True)
		#regressor = xgb.XGBRegressor(eta = 0.01, n_estimators = 1000, max_depth = 8)
		regressor.fit(X_train, Y_train.values.ravel())

		Y_predict = regressor.predict(X_test)
		print("TEST MSE for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, mean_squared_error(Y_test, Y_predict), predictionHoursAhead, lookBackHours))
		print("TEST R2 Score for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, r2_score(Y_test, Y_predict), predictionHoursAhead, lookBackHours))

		Y_predict = regressor.predict(X_train)
		print("TRAIN MSE for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, mean_squared_error(Y_train, Y_predict), predictionHoursAhead, lookBackHours))
		print("TRAIN R2 Score for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, r2_score(Y_train, Y_predict), predictionHoursAhead, lookBackHours))

#=====================================================================
def performPCAAnalysis(x, n):
	pca = PCA(n_components = n)
	pca_result = pca.fit_transform(x)

	#plt.plot(range(n), pca.explained_variance_ratio_)
	#plt.plot(range(n), np.cumsum(pca.explained_variance_ratio_))
	#plt.title("Component-wise and Cumulative Explained Variance")
	#plt.show()

	return pca_result



#======================================================================
def piecewiseSplit(featureSet, trainDays, predictDays, resetDays):
	train_set = []
	test_set = []

	t_hours = trainDays * 24
	p_hours = predictDays * 24
	r_hours = resetDays * 24
	cycle_hours = t_hours + p_hours + r_hours
	cycle_i = 0

	for index, current_row in featureSet.iterrows():
		if cycle_i <= t_hours:
			train_set.append(current_row)
			pass
		elif cycle_i > t_hours and cycle_i <= t_hours + p_hours:
			test_set.append(current_row)
			pass
		elif cycle_i > t_hours + p_hours and cycle_i < cycle_hours:
			pass
		else:
			cycle_i = 0
		cycle_i += 1

	return pd.DataFrame(train_set), pd.DataFrame(test_set)

#=====================================================================
# Parse model metadata out of the file name
# Example filename: featureset_is_clear_LocationCount3_LookBack2h_LookAhead24h.csv
def parseVarnameFname(fname):
	i = fname.index('_LocationCount')
	return fname[len('NOAA_featureset_') : i]

def parsePredictionIntervalFname(fname):
	i1 = fname.index('LookAhead')
	fname = fname [i1 : ]
	i2 = fname.index('h.csv')
	return fname[len('LookAhead') : i2]

def parseLookbackWindowFname(fname):
	i1 = fname.index('_LookBack')
	fname = fname[i1 : ]
	i2 = fname.index('h_LookAhead')
	return fname[len('_LookBack') : i2]
#=====================================================================

files = [
	'NOAA_featureset_WindSpeed_LocationCount9_LookBack3h_LookAhead12h.csv', 'NOAA_featureset_WindSpeed_LocationCount9_LookBack3h_LookAhead24h.csv',
	'NOAA_featureset__is_precip_LocationCount9_LookBack3h_LookAhead12h.csv', 'NOAA_featureset__is_precip_LocationCount9_LookBack3h_LookAhead24h.csv', 
	'NOAA_featureset__is_clear_LocationCount9_LookBack3h_LookAhead12h.csv', 'NOAA_featureset__is_clear_LocationCount9_LookBack3h_LookAhead24h.csv',
	'NOAA_featureset_Temp_LocationCount9_LookBack3h_LookAhead12h.csv', 'NOAA_featureset_Temp_LocationCount9_LookBack3h_LookAhead24h.csv']

for file in files:
		exploreModel(file, 50)
		print("=================\r\n")

