import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score

pd.set_option('display.max_rows', 250)

#=====================================================================
def exploreModel(targetLocationFile, adjacentLocationFiles, predictionHoursAhead, lookBackHours, predictedVariable, dependentVariables, isClassification):
	
	#(1) Merge the datasets
	merged_df = createMergedDataframe(targetLocationFile, adjacentLocationFiles)[22000:]

	#(2) Build the featureset (NOTE: this takes forever, begging for optimization)
	featureset = buildFeatureSet(merged_df, predictionHoursAhead, lookBackHours, dependentVariables, predictedVariable, len(adjacentLocationFiles))

	#(3) Train/Test Split
	Y = featureset[['Y']]
	X = featureset.drop(columns = ['Y', 'prediction_for_ts'])

	#(3a) (experimental): normalize data
	scaler = StandardScaler() 
	X = scaler.fit_transform(X)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

	
	if isClassification:
		regressor = MLPClassifier(hidden_layer_sizes = [int(len(featureset.columns) * .67), int(len(featureset.columns) * .1)])
		regressor.fit(X_train, Y_train.values.ravel())
		Y_predict = regressor.predict(X_test)
		score = f1_score(Y_test, Y_predict)
		print("Model Score for VAR = {} is {}".format(predictedVariable, score))

	else:
		regressor = MLPRegressor(hidden_layer_sizes = [int(len(featureset.columns) * .67), int(len(featureset.columns) * .1)])
		regressor.fit(X_train, Y_train.values.ravel())
		Y_predict = regressor.predict(X_test)
		score = r2_score(Y_test, Y_predict)	
		print("Model Score for VAR = {} Is {}".format(predictedVariable, score))
	
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
		if checkNHoursBeforeTsPresent(mergedDf, current_observation_ts - timedelta(hours = predictionHoursAhead), lookBackHours) and len(mergedDf.loc[current_observation_ts : current_observation_ts]) == 1:
			observations_included = observations_included + 1
			row_dict = buildFeatureRow(mergedDf, current_observation_ts, predictionHoursAhead, lookBackHours, featureVariables, predictedVariable, nAdjacentLocations)
			row_dicts.append(row_dict)
		else:
			observations_excluded = observations_excluded + 1


		current_observation_ts = current_observation_ts - timedelta(hours = 1)


	feature_set_df = pd.DataFrame(row_dicts)	
	print("Generated Feature Set, observations in = {}, out = {}".format(observations_included, observations_excluded))
	print("Feature Set feature count = {}".format(len(feature_set_df.columns)))

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
				# NOTE - a meh hack to not include day_of_year, hour_of_day cyclical vars multiple times..
				if (('_sin' not in feature_var) and('_cos' not in feature_var)):
					row_dict["{}_loc{}_{}h".format(feature_var, l, hour_num)] = current_row["{}{}".format(feature_var,l)]

		current_ts = current_ts + timedelta(hours = 1)
		hour_num = hour_num + 1

	#..and don't forget the predicted variable
	row_dict['Y'] = (mergedDf.loc[predictionTs])[predictedVariable]
			
	return row_dict
#=====================================================================
exploreModel('../Chicago_2011-2020_CLEANED.csv', 
	['../Madison_2011-2020_CLEANED.csv', '../GrandRapids_2011-2020_CLEANED.csv', '../GreenBay_2011-2020_CLEANED.csv', '../Des_Moines_2011-2020_CLEANED.csv',
	 '../Indianapolis_2011-2020_CLEANED.csv', '../Cincinatti_2011-2020_CLEANED.csv', '../Toronto_2011-2020_CLEANED.csv', '../StLouis_2011-2020_CLEANED.csv',
	 '../Minneapolis_2011-2020_CLEANED.csv', '../Cleveland_2011-2020_CLEANED.csv', '../Columbus_2011-2020_CLEANED.csv', '../Sault_Ste_Marie_2011-2020_CLEANED.csv'], 
	48, 
	8, 
	'is_precip',
	['Temp', 'WindSpeed', 'WindNortherly', 'WindEasterly', 'day_of_year_sin', 'day_of_year_cos', 'hour_of_day_sin', 'hour_of_day_cos', 'Pressure', 'Humidity', 'DewPoint', 'is_clear', 'is_precip', 'is_heavy_precip', 'is_tstorm'],
	True)


# PROBLEM: it appears that precipitation is not reflected in Condition until about March 2014!
# ..therefore, we may need to consider sources other than WUNDERGROUND :(
#
##################################################################################
# PRELIMINARY RESULTS for predicting weather characteristics for a given hour
#
# Data after March 2014, 8h prediction lookback, Neural Network; 
##################################################################################
#	Resulting Model Scores:
#		is_clear =  70% (12h, 8 loc)	67% (12h, 4 loc)	73% (24h, 12 loc)	68%(48h, 8 loc)		68%(48h, 12 loc)	66% (96h, 8 loc)	64% (144h, 8 loc)
#		is_precip = 55% (12h, 8 loc)	52% (12h, 4 loc)	57% (24h, 12 loc)	61%(48h, 8 loc)		53%(48h, 12 loc)	54% (96h, 8 loc)	51% (144h, 8 loc)
#		WindSpeed = 49% (12h, 8 loc)	5%  (12h, 4 loc)	55% (24h, 12 loc)	45%(48h, 8 loc)		52%(48h, 12 loc)
#		Temp = 		96% (12h, 8 loc)	91%  (12h, 4 loc) 	96% (24h, 8 loc)	92%(48h, 8 loc)							92% (96h, 8 loc)	87% (144h, 8 loc)
#####################################################################################
#
#  Adding locations has the most effect on Wind prediction: e.g. 24h/12loc gives us 55% as opposed to 49% for 24h/8loc
#  On the other hand, is_precip and is_clear are either unaffected by extra locations or affected negatively
#