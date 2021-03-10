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

pd.set_option('display.max_rows', 250)

#=====================================================================
def exploreModel(featuresetFile):

	predictedVariable = parseVarnameFname(featuresetFile)
	predictionHoursAhead = parsePredictionIntervalFname(featuresetFile)
	lookBackHours = parseLookbackWindowFname(featuresetFile)
	
	featureset = pd.read_csv(featuresetFile, parse_dates=['prediction_for_ts'])
	featureset = featureset.dropna()

	print("Loaded file for VAR = {}, predictionInterval = {}h, lookback = {}h".format(predictedVariable, predictionHoursAhead, lookBackHours))
	#columns_to_drop = []
	#for col in featureset.columns:
	#	if 'loc0' not in col and 'loc1' not in col and 'loc2' not in col and 'loc3' not in col and col != 'Y' and col != 'prediction_for_ts':
	#		columns_to_drop.append(col)
	#featureset = featureset.drop(columns = columns_to_drop)
	#print("Dropped {} columns, {} remaining".format(len(columns_to_drop), len(featureset.columns)))
	
	#coll = featureset.corr()[['Y']].sort_values('Y')
	#for i, row in coll.iterrows():
	#	if abs(row['Y']) > 0.2:
	#		print(row)
	#return

	#(3) Train/Test Split
	featureset_train, featureset_test = piecewiseSplit(featureset, 8, 2, 3)
	print("Piecewise Split: trainset len = {}, testset_len = {}".format(len(featureset_train), len(featureset_test)))

	Y_train = featureset_train[['Y']]
	X_train = featureset_train.drop(columns = ['Y', 'prediction_for_ts'])

	Y_test = featureset_test[['Y']]
	X_test = featureset_test.drop(columns = ['Y', 'prediction_for_ts'])

	#(4) normalize data
	#scaler = MinMaxScaler() 
	#X_train = scaler.fit_transform(X_train)
	#X_test = scaler.fit_transform(X_test)

	if "is_" in predictedVariable: # Classification
		#regressor = MLPClassifier(hidden_layer_sizes = [int(len(featureset.columns) * .7), int(len(featureset.columns) * .3)], alpha = 0.1)
		regressor = xgb.XGBClassifier(eta = 0.05, n_estimators = 4000, max_depth = 10, eval_metric = 'error')
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
		#regressor = MLPRegressor(hidden_layer_sizes = [int(len(featureset.columns) * .8), int(len(featureset.columns) * .4)], alpha = 0.1)
		regressor = xgb.XGBRegressor(eta = 0.01, n_estimators = 1000, max_depth = 8)
		regressor.fit(X_train, Y_train.values.ravel())

		Y_predict = regressor.predict(X_test)
		print("TEST MSE for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, mean_squared_error(Y_test, Y_predict), predictionHoursAhead, lookBackHours))
		print("TEST R2 Score for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, r2_score(Y_test, Y_predict), predictionHoursAhead, lookBackHours))

		Y_predict = regressor.predict(X_train)
		print("TRAIN MSE for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, mean_squared_error(Y_train, Y_predict), predictionHoursAhead, lookBackHours))
		print("TRAIN R2 Score for VAR = {} Is {} for prediction interval = {}h and for lookback window = {}h".format(predictedVariable, r2_score(Y_train, Y_predict), predictionHoursAhead, lookBackHours))

#=====================================================================
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
	return fname[len('featureset_') : i]

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

files = ['featureset_is_precip_LocationCount8_LookBack2h_LookAhead8h.csv']

for file in files:
		exploreModel(file)
		print("=================\r\n")


