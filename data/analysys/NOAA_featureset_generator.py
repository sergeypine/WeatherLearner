import pandas as pd
import numpy as np
from datetime import datetime, timedelta

pd.set_option('display.max_rows', 250)

#=====================================================================
def saveFeatureSet(targetLocationFile, adjacentLocationFiles, predictionHoursAhead, lookBackHours, predictedVariable, variablesForLookBack, variablesToAggregate):
	
	#(1) Merge the datasets
	merged_df = createMergedDataframe(targetLocationFile, adjacentLocationFiles)

	#(2) Build the featureset (NOTE: this takes forever, begging for optimization)
	featureset = buildFeatureSet(merged_df, predictionHoursAhead, lookBackHours, variablesForLookBack, variablesToAggregate, predictedVariable, len(adjacentLocationFiles))

	featureset.to_csv("NOAA_featureset_{}_LocationCount{}_LookBack{}h_LookAhead{}h.csv".format(predictedVariable,  len(adjacentLocationFiles), lookBackHours, predictionHoursAhead), index = False)

#=====================================================================
def createMergedDataframe(targetLocationFile, adjacentLocationFiles):
	target_df = pd.read_csv(targetLocationFile, parse_dates=['DATE'])
	merged_df = target_df
	suffix_no = 1
	for adjacentLocationFile in adjacentLocationFiles:
		adjacent_df = pd.read_csv(adjacentLocationFile, parse_dates=['DATE'])
		
		#Take control of column name suffix in the dataset being merged in
		adjacent_df = adjacent_df.add_suffix(str(suffix_no))
		adjacent_df = adjacent_df.rename(columns = {"DATE{}".format(suffix_no) :'DATE'})
		merged_df = pd.merge(merged_df, adjacent_df, on='DATE')
		suffix_no = suffix_no + 1

	print("Merged Location Data; initial rows = {}, final rows = {}".format(len(target_df), len(merged_df)))
	
	merged_df = merged_df.set_index('DATE', drop=False)
	return merged_df

#=====================================================================
def buildFeatureSet(mergedDf, predictionHoursAhead, lookBackHours, variablesForLookBack, variablesToAggregate, predictedVariable, nAdjacentLocations):
	earliest_observation_ts = mergedDf['DATE'].min()
	current_observation_ts = mergedDf['DATE'].max()

	row_dicts = []
	loop_guard = 0
	observations_included = 0

	#iterate from the last iteration backwards
	while(current_observation_ts - timedelta(hours=1) >= earliest_observation_ts):
		# OPTIMIZATION: pair down the DF to cover the vicinity time window only
		timeVicinityDf = mergedDf[current_observation_ts - timedelta(days = 4) : current_observation_ts + timedelta(hours = 6)]

		row_dict = buildFeatureRow(timeVicinityDf, current_observation_ts, predictionHoursAhead, lookBackHours, variablesForLookBack, variablesToAggregate, predictedVariable, nAdjacentLocations)
		row_dicts.append(row_dict)

		current_observation_ts = current_observation_ts - timedelta(hours = 1)
		observations_included = observations_included + 1
		
		#loop_guard = loop_guard + 1
		#if loop_guard > 10:
		#	break

		print("...processing for TS = {}".format(current_observation_ts))

	feature_set_df = pd.DataFrame(row_dicts)	
	print("Generated Feature Set, observations included = {}".format(observations_included))
	print("Feature Set feature count = {}".format(len(feature_set_df.columns)))

	return feature_set_df

#=====================================================================
def buildFeatureRow(mergedDf, predictionTs, predictionHoursAhead, lookBackHours, featureVariables, variablesToAggregate, predictedVariable, nAdjacentLocations):

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
				if (('day_sin' not in feature_var) and('day_cos' not in feature_var) and ('hour_sin' not in feature_var) and ('hour_cos' not in feature_var)):
					row_dict[generateFeaturesetFeatureName(feature_var, l, hour_num)] = current_row["{}{}".format(feature_var,l)]

		hour_num = hour_num - 1
	# Capture longer running aggregations
	# TODO -  (need to figure out how to speed up Pandas...)
	for hrs in [6, 18]:

		current_ts = predictionTs - timedelta(hours = hrs + predictionHoursAhead)
		end_ts = predictionTs - timedelta(hours = predictionHoursAhead)
		subsetDf = mergedDf.loc[current_ts : end_ts]

		# For target Location
		for agg_var in variablesToAggregate:
			setStatsLastNhours(subsetDf, hrs, row_dict, agg_var, 0)

		# For adjacent locations
		for l in range(1, nAdjacentLocations + 1):
			for agg_var in variablesToAggregate:
				setStatsLastNhours(subsetDf, hrs, row_dict, agg_var, l)		

	#..and don't forget the predicted variable
	setPredictedVar(mergedDf, row_dict, predictionTs, predictedVariable)
			
	return row_dict
#=====================================================================
def generateFeaturesetFeatureName(featureName, locN, hoursBack):
	return "{}_loc{}_{}h".format(featureName, locN, hoursBack)

#=====================================================================


# Use smoothing for target VAR values for some vars: instead of the particular hour, consider a time interval surrounding it
def setPredictedVar(mergedDf, rowDict, predictionTs, predictedVariable):

	# Use wider interval for clouds and precipitation	
	half_interval = 1
	if predictedVariable == '_is_clear' or predictedVariable == '_is_precip':
		half_interval = 3

	current_ts = predictionTs - timedelta(hours = half_interval)
	end_ts = predictionTs + timedelta(hours = half_interval)
	subsetDf = mergedDf.loc[current_ts : end_ts]
	
	# accumulate all values over that interval
	interval_vals = subsetDf[predictedVariable].values.tolist()
	
	# Precipitation: if it precipitated during a single hour during the period, return yes
	if predictedVariable == '_is_precip':
		if 1 in interval_vals:
			rowDict['Y'] = 1
		else:
			rowDict['Y'] = 0

	#is_clear: if it was not clear during a single hour during the period, return no
	elif predictedVariable == '_is_clear':
		if 0 in interval_vals:
			rowDict['Y'] = 0
		else:
			rowDict['Y'] = 1
	# For Wind and Temp just return the average
	else:
		rowDict['Y'] = np.mean(interval_vals)


#=====================================================================
def setStatsLastNhours(subsetDf, nHours, rowDict, averagedVar, locN):

	var_name = averagedVar
	if (locN != 0):
		var_name = "{}{}".format(averagedVar, locN)

	interval_vals = subsetDf[var_name]

	rowDict["{}_loc{}_avg_{}h".format(averagedVar, locN, nHours)] = np.mean(interval_vals)
	rowDict["{}_loc{}_min_{}h".format(averagedVar, locN, nHours)] = np.min(interval_vals)
	rowDict["{}_loc{}_max_{}h".format(averagedVar, locN, nHours)] = np.max(interval_vals)

#=====================================================================

for predicted_var in ['WindSpeed']:
	for prediction_interval in [12, 24]:
		for lookback_window in [3]:
			saveFeatureSet(
				'../NOAA/noaa_2011-2020_chicago_PREPROC.csv',
				['../NOAA/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../NOAA/noaa_2011-2020_des-moines_PREPROC.csv', '../NOAA/noaa_2011-2020_quincy_PREPROC.csv', '../NOAA/noaa_2011-2020_rochester_PREPROC.csv', '../NOAA/noaa_2011-2020_madison_PREPROC.csv', 
				'../NOAA/noaa_2011-2020_st-louis_PREPROC.csv', '../NOAA/noaa_2011-2020_indianapolis_PREPROC.csv',  '../NOAA/noaa_2011-2020_lansing_PREPROC.csv',  '../NOAA/noaa_2011-2020_green-bay_PREPROC.csv'],
				prediction_interval, 
				lookback_window, 
				predicted_var,
				['_day_sin', '_day_cos', '_hour_sin', '_hour_cos',  '_wind_dir_sin', '_wind_dir_cos',  'PressureChange'],
				['_cloud_intensity', 'Precipitation', '_is_thunder', '_is_snow', 'Temp', 'DewPoint', 'Humidity', 'Pressure', 'CloudAltitude', 'WindSpeed', 'WindGust']
			)

			print("=================\r\n")


