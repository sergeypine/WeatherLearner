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
def saveFeatureSet(targetLocationFile, adjacentLocationFiles, predictionHoursAhead, lookBackHours, predictedVariable, dependentVariables, isClassification):
	
	#(1) Merge the datasets
	merged_df = createMergedDataframe(targetLocationFile, adjacentLocationFiles)

	#(2) Build the featureset (NOTE: this takes forever, begging for optimization)
	featureset = buildFeatureSet(merged_df, predictionHoursAhead, lookBackHours, dependentVariables, predictedVariable, len(adjacentLocationFiles))

	featureset.to_csv("featureset_{}_LocationCount{}_LookBack{}h_LookAhead{}h.csv".format(predictedVariable,  len(adjacentLocationFiles), lookBackHours, predictionHoursAhead), index = False)

#=====================================================================
def createMergedDataframe(targetLocationFile, adjacentLocationFiles):
	target_df = pd.read_csv(targetLocationFile, parse_dates=['Time'])
	merged_df = target_df
	suffix_no = 1
	for adjacentLocationFile in adjacentLocationFiles:
		adjacent_df = pd.read_csv(adjacentLocationFile, parse_dates=['Time'])
		
		#Take control of column name suffix in the dataset being merged in
		adjacent_df = adjacent_df.add_suffix(str(suffix_no))
		adjacent_df = adjacent_df.rename(columns = {"Time{}".format(suffix_no) :'Time'})
		merged_df = pd.merge(merged_df, adjacent_df, on='Time')
		suffix_no = suffix_no + 1

	print("Merged Location Data; initial rows = {}, final rows = {}".format(len(target_df), len(merged_df)))
	
	merged_df = merged_df.set_index('Time', drop=False)
	return merged_df

#=====================================================================
def buildFeatureSet(mergedDf, predictionHoursAhead, lookBackHours, featureVariables, predictedVariable, nAdjacentLocations):
	earliest_observation_ts = mergedDf['Time'].min()
	current_observation_ts = mergedDf['Time'].max()

	observations_included = 0
	observations_excluded = 0

	row_dicts = []
	#iterate from the last iteration backwards
	while(current_observation_ts - timedelta(hours=1) >= earliest_observation_ts):
		# OPTIMIZATION: pair down the DF to cover the vicinity time window only
		timeVicinityDf = mergedDf[current_observation_ts - timedelta(days = 4) : current_observation_ts + timedelta(hours = 6)]

		if checkNHoursBeforeTsPresent(timeVicinityDf, current_observation_ts - timedelta(hours = predictionHoursAhead), lookBackHours) and len(timeVicinityDf.loc[current_observation_ts : current_observation_ts]) == 1:
			observations_included = observations_included + 1
			row_dict = buildFeatureRow(timeVicinityDf, current_observation_ts, predictionHoursAhead, lookBackHours, featureVariables, predictedVariable, nAdjacentLocations)
			row_dicts.append(row_dict)
		else:
			observations_excluded = observations_excluded + 1


		current_observation_ts = current_observation_ts - timedelta(hours = 1)
		#print("...processing for TS = {}".format(current_observation_ts))

	feature_set_df = pd.DataFrame(row_dicts)	
	print("Generated Feature Set, observations in = {}, out = {}".format(observations_included, observations_excluded))
	print("Feature Set feature count = {}".format(len(feature_set_df.columns)))

	return feature_set_df

#=====================================================================
def checkNHoursBeforeTsPresent(mergedDf, ts, hourCount):
	return len(mergedDf.loc[ts - timedelta(hours = hourCount - 1) : ts]) == hourCount

#=====================================================================
def buildFeatureRow(mergedDf, predictionTs, predictionHoursAhead, lookBackHours, featureVariables, predictedVariable, nAdjacentLocations):
	current_ts = predictionTs - timedelta(hours = predictionHoursAhead + lookBackHours)
	end_ts = predictionTs - timedelta(hours = predictionHoursAhead)
	hour_num = lookBackHours
	row_dict = {}

	row_dict['prediction_for_ts'] = predictionTs
	lookBackDf = mergedDf.loc[current_ts : end_ts]

	# Capture Detailed Data during Lookback Window for all locations
	for index, current_row in lookBackDf.iterrows():

		# Add Target Location features
		for feature_var in featureVariables:
			row_dict[generateFeaturesetFeatureName(feature_var, 0, hour_num)] = current_row[feature_var]

		# Add Adjacent Locations features
		for l in range(1, nAdjacentLocations + 1):
			for feature_var in featureVariables:
				# NOTE - a meh hack to not include day_of_year, hour_of_day cyclical vars multiple times..
				if (('_sin' not in feature_var) and('_cos' not in feature_var)):
					row_dict[generateFeaturesetFeatureName(feature_var, l, hour_num)] = current_row["{}{}".format(feature_var,l)]

		hour_num = hour_num - 1
	
	#########################################################################	
	# Capture longer running aggregations
	# TODO -  (need to figure out how to speed up Pandas...)
	#for hrs in [12, 24, 48, 72]:

	#	current_ts = predictionTs - timedelta(hours = hrs)
	#	end_ts = predictionTs
	#	subsetDf = mergedDf.loc[current_ts : end_ts]

	#	setStatsLastNhours(subsetDf, hrs, row_dict, 'precip_intensity')
	#	setStatsLastNhours(subsetDf, hrs, row_dict, 'cloud_intensity')
	#	setStatsLastNhours(subsetDf, hrs, row_dict, 'WindNortherly')
	#	setStatsLastNhours(subsetDf, hrs, row_dict, 'WindEasterly')
	#########################################################################

	#..and don't forget the predicted variable
	setPredictedVar(mergedDf, row_dict, predictionTs, predictedVariable)
			
	return row_dict
#=====================================================================
def generateFeaturesetFeatureName(featureName, locN, hoursBack):
	return "{}_loc{}_{}h".format(featureName, locN, hoursBack)

#=====================================================================


# Use smoothing for target VAR values for some vars: instead of the particular hour, consider a time interval surrounding it
def setPredictedVar(mergedDf, rowDict, predictionTs, predictedVariable):
	
	#6h interval
	current_ts = predictionTs - timedelta(hours = 2)
	end_ts = predictionTs + timedelta(hours = 3)
	subsetDf = mergedDf.loc[current_ts : end_ts]
	
	# accumulate all values over that interval
	interval_vals = subsetDf[predictedVariable].values.tolist()
	
	# Precipitation: if it rained during a single hour during the period, return yes
	if predictedVariable == 'is_precip':
		if 1 in interval_vals:
			rowDict['Y'] = 1
		else:
			rowDict['Y'] = 0

	#is_clear: if it was not clear during a single hour during the period, return no
	elif predictedVariable == 'is_clear':
		if 0 in interval_vals:
			rowDict['Y'] = 0
		else:
			rowDict['Y'] = 1
	# For Wind and Temp just return the average
	else:
		rowDict['Y'] = np.mean(interval_vals)


#=====================================================================
def setStatsLastNhours(subsetDf, nHours, rowDict, averagedVar):

	interval_vals = subsetDf[[averagedVar]]

	rowDict["{}_avg_{}h".format(averagedVar, nHours)] = np.mean(interval_vals)
	rowDict["{}_std_{}h".format(averagedVar, nHours)] = np.std(interval_vals)

#=====================================================================

for predicted_var in ['is_clear', 'is_precip', 'Temp', 'WindSpeed']:
	for prediction_interval in [24]:
		for lookback_window in [2]:
			saveFeatureSet('../Chicago_2011-2020_CLEANED.csv', 
				#['../Peoria_2011-2020_CLEANED.csv', '../CedarRapids_2011-2020_CLEANED.csv', '../Madison_2011-2020_CLEANED.csv', '../GreenBay_2011-2020_CLEANED.csv', 
				# '../GrandRapids_2011-2020_CLEANED.csv', '../FortWayne_2011-2020_CLEANED.csv', '../Indianapolis_2011-2020_CLEANED.csv'],
				['../Peoria_2011-2020_CLEANED.csv', '../CedarRapids_2011-2020_CLEANED.csv', '../Madison_2011-2020_CLEANED.csv'],
				prediction_interval, 
				lookback_window, 
				predicted_var,
				['day_of_year_sin', 'day_of_year_cos', 'hour_of_day_cos', 'hour_of_day_sin', 'Temp', 'WindSpeed', 'WindNortherly', 'WindEasterly', 'is_var', 'Pressure', 'Humidity', 'DewPoint', 'cloud_intensity', 'precip_intensity'],
				(True if ('is_' in predicted_var) else False)
			)

			print("=================\r\n")


