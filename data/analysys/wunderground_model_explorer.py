import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score

pd.set_option('display.max_rows', 250)

#=====================================================================
def exploreModel(targetLocationFile, adjacentLocationFiles, predictionHoursAhead, lookBackHours, predictedVariable, dependentVariables, isClassification):
	
	#(1) Merge the datasets
	merged_df = createMergedDataframe(targetLocationFile, adjacentLocationFiles)

	#(2) Build the featureset
	featureset = buildFeatureSet(merged_df, predictionHoursAhead, lookBackHours, dependentVariables, predictedVariable, len(adjacentLocationFiles))

	#(3) Train/Test Split
	Y = featureset[['Y']]
	X = featureset.drop(columns = ['Y', 'prediction_for_ts'])
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
	
	if isClassification:
		regressor = LogisticRegression(solver='liblinear')
		regressor.fit(X_train, Y_train)
		Y_predict = regressor.predict(X_test)
		score = f1_score(Y_test, Y_predict)
		print("Model Score is {}".format(score))

	else:
		regressor = LinearRegression()
		regressor.fit(X_train, Y_train)
		Y_predict = regressor.predict(X_test)
		score = r2_score(Y_test, Y_predict)	
		print("Model Score Is {}".format(score))
	
#=====================================================================
def createMergedDataframe(targetLocationFile, adjacentLocationFiles):
	target_df = pd.read_csv(targetLocationFile, parse_dates=['Time'])
	merged_df = target_df
	for adjacentLocationFile in adjacentLocationFiles:
		adjacent_df = pd.read_csv(adjacentLocationFile, parse_dates=['Time'])
		merged_df = pd.merge(merged_df, adjacent_df, on='Time', suffixes = ("1", "2"))
	print("Merged Location Data; initial rows = {}, final rows = {}".format(len(target_df), len(merged_df)))
	merged_df = merged_df.set_index('Time', drop=False)
	print(merged_df.columns)
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
		if checkNHoursBeforeTsPresent(mergedDf, current_observation_ts - timedelta(hours = predictionHoursAhead), lookBackHours) and len(mergedDf.loc[current_observation_ts : current_observation_ts]) == 1:
			observations_included = observations_included + 1
			row_dict = buildFeatureRow(mergedDf, current_observation_ts, predictionHoursAhead, lookBackHours, featureVariables, predictedVariable, nAdjacentLocations)
			row_dicts.append(row_dict)
		else:
			observations_excluded = observations_excluded + 1


		current_observation_ts = current_observation_ts - timedelta(hours = 1)

	print("Generated Feature Set, observations in = {}, out = {}".format(observations_included, observations_excluded))
	feature_set_df = pd.DataFrame(row_dicts)
	return feature_set_df

#=====================================================================
def checkNHoursBeforeTsPresent(mergedDf, ts, hourCount):
	return len(mergedDf.loc[ts - timedelta(hours = hourCount - 1) : ts]) == hourCount

#=====================================================================
def buildFeatureRow(mergedDf, predictionTs, predictionHoursAhead, lookBackHours, featureVariables, predictedVariable, nAdjacentLocations):
	current_ts = predictionTs - timedelta(hours = predictionHoursAhead + lookBackHours - 1)
	end_ts = predictionTs - timedelta(hours = predictionHoursAhead)
	hour_num = 1
	row_dict = {}

	row_dict['prediction_for_ts'] = predictionTs

	while current_ts <= end_ts:
		current_row = mergedDf.loc[current_ts]
		
		# Add Target Location features
		for feature_var in featureVariables:
			row_dict["{}_loc{}_{}h".format(feature_var, 0, hour_num)] = current_row[feature_var]

		# Add Adjacent Locations features
		for l in range(1, nAdjacentLocations + 1):
			for feature_var in featureVariables:
				# NOTE - a meh hack to not include day_of_year, hour_of_day cyclical vars twice..
				if (('_sin' not in feature_var) and('_cos' not in feature_var)):
					row_dict["{}_loc{}_{}h".format(feature_var, l, hour_num)] = current_row["{}{}".format(feature_var,l)]

		current_ts = current_ts + timedelta(hours = 1)
		hour_num = hour_num + 1

	#..and don't forget the predicted variable
	row_dict['Y'] = (mergedDf.loc[predictionTs])[predictedVariable]
			
	return row_dict
#=====================================================================
exploreModel('../Chicago_2011-2020_CLEANED.csv', 
	['../Madison_2011-2020_CLEANED.csv', '../GrandRapids_2011-2020_CLEANED.csv'], 
	24, 
	6, 
	'is_tstorm',
	['Temp', 'WindSpeed', 'WindNortherly', 'WindEasterly', 'day_of_year_sin', 'day_of_year_cos', 'hour_of_day_sin', 'hour_of_day_cos', 'Pressure', 'is_precip', 'is_clear', 'Humidity'],
	True)