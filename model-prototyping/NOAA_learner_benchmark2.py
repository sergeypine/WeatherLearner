#########################################################################################################
# NOTE: Only features/locations selected by a one-by-one feature selection process are used 
########################################################################################################

import pandas as pd
import numpy as np
import math
from keras.models import Sequential
from keras.layers import LSTM,Dense ,Dropout
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 250)

#=====================================================================
def exploreModel(mission):

	# Load the Target Location and Adjacent Locations Datasets and drop the unused columns
	featureset = buildFeatureSet(mission.targetLocation, mission.adjacentLocations, mission.predictedVariable, mission.featuresToUse)

	# Split the data: 6 years for training, 2 for validation & 2 for testing
	n = len(featureset)
	train_df = featureset[0 : int(n*0.70)]
	val_df = featureset[int(n*0.70) : ]

	# Normalize input data (NOTE: TF tutorial also scales the target variable)
	train_df, val_df = normalizeData(train_df, val_df,  mission.predictedVariable, mission.featuresToUse, len(mission.adjacentLocations))

	agg_interval = 2 * mission.aggHalfInterval + 1
	wg = WindowGenerator(input_width=mission.lookBackHours, 
						 label_width = agg_interval, 
						 shift= mission.lookAheadHrs - mission.aggHalfInterval, 
						 label_columns=[mission.predictedVariable], 
						 train_df = train_df, val_df = val_df, test_df = None)
		


	# Build the desired model
	inputCnt = (len(mission.adjacentLocations) + 1) * (len(mission.featuresToUse) + 1) * mission.lookBackHours
	print("Input Cnt = {}".format(inputCnt))
	model = None
	if mission.modelToUse == 'LINEAR':
		model = buildLinearModel("is_" in mission.predictedVariable, agg_interval, inputCnt)

	if mission.modelToUse == 'NN':
		model = buildSimpleNNModel("is_" in mission.predictedVariable, agg_interval, inputCnt)

	if mission.modelToUse == 'DNN':
		model = buildDNNModel("is_" in mission.predictedVariable, agg_interval, inputCnt)
	
	if mission.modelToUse == 'CNN':
		model = buildConvModel("is_" in mission.predictedVariable, mission.lookBackHours, agg_interval, inputCnt)
	
	if mission.modelToUse == 'RNN':
		model = buildLSTMModel("is_" in mission.predictedVariable, agg_interval, inputCnt)

	# Configure early stopping so we don't learn the training data too well at the expense of test/validation data (overfit)
	#	(use different criteria for Classification and Regression)
	if "is_" in mission.predictedVariable:
		esCallback = tf.keras.callbacks.EarlyStopping(monitor='val_recall', mode="max", patience=20, min_delta = 0.0002, restore_best_weights = True)
	else:
		esCallback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', mode="min", patience=20, min_delta = 0.02, restore_best_weights = True)

	# NOTE: provide the Validation Dataset so that the Model does not check itself on Train	
	model.fit(wg.train, validation_data = wg.val, callbacks = [esCallback], epochs = 100)	

	eval_func = evaluateClassificationModel if "is_" in mission.predictedVariable else evaluateRegressionModel
	
	#_is_clear variable has a special aggregation rule
	options = []
	if mission.predictedVariable == '_is_clear':
		options.append("TrueIfAllTrue")

	model.save("saved_models/{}_{}h_{}.h5".format(mission.predictedVariable, mission.lookAheadHrs, mission.modelToUse))

	print("\r\n- Performance on *TRAINING* data ({}):".format(mission))
	eval_func(model, wg.train, options)
	print("\r\n- Performance on *TEST* data ({}):".format(mission))
	return eval_func(model, wg.val, options)

#======================================================================
def buildFeatureSet(targetLocationFile, adjacentLocationFiles, predictedVariable, featuresToUse):
	target_df = pd.read_csv(targetLocationFile, parse_dates=['DATE'])
	target_df = dropUnusedColumns(target_df, predictedVariable, featuresToUse)
	merged_df = target_df
	suffix_no = 1
	
	# Merge adjacent location files one by one relying on DATA
	for adjacentLocationFile in adjacentLocationFiles:
		adjacent_df = pd.read_csv(adjacentLocationFile, parse_dates=['DATE'])
		adjacent_df = dropUnusedColumns(adjacent_df, predictedVariable, featuresToUse)
		
		#Take control of column name suffix in the dataset being merged in
		adjacent_df = adjacent_df.add_suffix(str(suffix_no))
		adjacent_df = adjacent_df.rename(columns = {"DATE{}".format(suffix_no) :'DATE'})
		merged_df = pd.merge(merged_df, adjacent_df, on='DATE')
		suffix_no = suffix_no + 1

	# DATA column is of no use in the modelling stage
	merged_df = merged_df.drop(columns=['DATE'])
	return merged_df

#======================================================================
def aggregateQuantities(df, ahiDict):
	df = df.reset_index()

	return df

#======================================================================
def dropUnusedColumns(df, predictedVariable, featuresToUse):
	all_columns = featuresToUse.copy()
	all_columns.append('DATE')
	all_columns.append(predictedVariable)
	df = df[all_columns]

	return df

#======================================================================
def normalizeData(trainDf, valDf, predictedVariable, featuresToUse, adjacentLocationCount):

	columns_to_normalize = featuresToUse.copy()
	
	prefixes_to_normalize = featuresToUse.copy()
	prefixes_to_normalize.append(predictedVariable)
	for loc in range(1, 1 + adjacentLocationCount):
		for prefix in prefixes_to_normalize:
			columns_to_normalize.append("{}{}".format(prefix, loc))

	# Normalize input data (NOTE: TF tutorial also scales the target variable)
	train_mean = trainDf[columns_to_normalize].mean()
	train_std = trainDf[columns_to_normalize].std()

	trainDf[columns_to_normalize] = (trainDf[columns_to_normalize] - train_mean) / train_std
	valDf[columns_to_normalize] = (valDf[columns_to_normalize] - train_mean) / train_std

	return trainDf, valDf


#======================================================================		
#======================================================================
# 	BEGIN Data reformatting, taken from https://www.tensorflow.org/tutorials/structured_data/time_series
#======================================================================
#======================================================================
class WindowGenerator():

	def __init__(self, 
		input_width, # Lookback Window (hours into the past to base predictions on)
		label_width, # Aggregation Interval (how many hours of data we'll be predicting)
		shift, # How many hours in advance we'll be predicting
	    train_df, val_df, test_df, # Training, Validation and Testing sets
	    label_columns=None):
		
		# Store the raw data.
		self.train_df = train_df
		self.val_df = val_df
		self.test_df = test_df

		# Work out the label column indices.
		self.label_columns = label_columns
		if label_columns is not None:
		  self.label_columns_indices = {name: i for i, name in
		                                enumerate(label_columns)}
		self.column_indices = {name: i for i, name in
		                       enumerate(train_df.columns)}

		# Work out the window parameters.
		self.input_width = input_width
		self.label_width = label_width
		self.shift = shift

		self.total_window_size = input_width + shift

		self.input_slice = slice(0, input_width)
		self.input_indices = np.arange(self.total_window_size)[self.input_slice]

		self.label_start = self.total_window_size - self.label_width
		self.labels_slice = slice(self.label_start, None)
		self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

	def __repr__(self):
		return '\n'.join([
		    f'Total window size: {self.total_window_size}',
		    f'Input indices: {self.input_indices}',
		    f'Label indices: {self.label_indices}',
		    f'Label column name(s): {self.label_columns}'])

	def split_window(self, features):
		inputs = features[:, self.input_slice, :]
		labels = features[:, self.labels_slice, :]
		if self.label_columns is not None:
			labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

		# Slicing doesn't preserve static shape information, so set the shapes
		# manually. This way the `tf.data.Datasets` are easier to inspect.
		inputs.set_shape([None, self.input_width, None])
		labels.set_shape([None, self.label_width, None])

		return inputs, labels

	def make_dataset(self, data):
		data = np.array(data, dtype=np.float32)
		ds = tf.keras.preprocessing.timeseries_dataset_from_array(
		  data=data,
		  targets=None,
		  sequence_length=self.total_window_size,
		  sequence_stride=1,
		  shuffle=False,
		  batch_size=32,)

		ds = ds.map(self.split_window)

		return ds
	
	@property
	def train(self):
		return self.make_dataset(self.train_df)

	@property
	def val(self):
		return self.make_dataset(self.val_df)

	@property
	def test(self):
		return self.make_dataset(self.test_df)

	@property
	def example(self):
		"""Get and cache an example batch of `inputs, labels` for plotting."""
		result = getattr(self, '_example', None)
		if result is None:
			# No example batch was found, so get one from the `.train` dataset
			result = next(iter(self.train))
			# And cache it for next time
			self._example = result
		return result

#======================================================================		
#======================================================================
# 	END Data reformatting, taken from https://www.tensorflow.org/tutorials/structured_data/time_series
#======================================================================
#======================================================================

def evaluateClassificationModel(model, testSet, options = []):

	predicted_labels =(model.predict(testSet, verbose = 1) > 0.5).astype("int32")
	true_labels = np.concatenate([y for x, y in testSet], axis=0)

	assert len(predicted_labels) == len (true_labels)

	# We are forecasting for a number of hours: 
	# aggregate each forecast series using the "True iff 1 or more is True" rule (default) or
	# "True iff all True" (option specified)
	predicted_agg = []
	true_agg = []
	for i in range(0, len(predicted_labels)):
		predicted_i = predicted_labels[i].flatten()
		true_i = true_labels[i].flatten()

		if not "TrueIfAllTrue" in options:
			predicted_i_agg = 1 if sum(predicted_i) > 0 else 0
			true_i_agg = 1 if sum(true_i) > 0 else 0
		else:
			predicted_i_agg = 1 if sum(predicted_i) == len(predicted_i) else 0
			true_i_agg = 1 if sum(true_i) == len(true_i) else 0

		predicted_agg.append(predicted_i_agg)
		true_agg.append(true_i_agg)

	recall = truncate(recall_score(true_agg, predicted_agg), 2)
	precision = truncate(precision_score(true_agg, predicted_agg), 2)
	f1 = truncate(f1_score(true_agg, predicted_agg), 2)
	mcc = truncate(matthews_corrcoef(true_agg, predicted_agg), 2)

	print(confusion_matrix(true_agg, predicted_agg))
	print("Recall = {}, Precision = {}, F1 = {}, MCC = {}".format(recall, precision, f1, mcc))

	return {
		"Recall": recall,
		"Precision": precision, 
		"F1:" : f1,
		"MCC:": mcc
	}	

def evaluateRegressionModel(model, testSet, options=[]):
	predicted_values = model.predict(testSet)
	true_values = np.concatenate([y for x, y in testSet], axis=0)

	assert len(predicted_values) == len (true_values)

	predicted_agg = []
	true_agg = []
	for i in range(0, len(predicted_values)):
		predicted_i = predicted_values[i].flatten()
		true_i = true_values[i].flatten()

		predicted_i_agg, true_i_agg = np.mean(predicted_i), np.mean(true_i)
		predicted_agg.append(predicted_i_agg)
		true_agg.append(true_i_agg)

	rmse = truncate(math.sqrt(mean_squared_error(true_agg, predicted_agg)), 2)
	mae = truncate(mean_absolute_error(true_agg, predicted_agg), 2)
	r2 = truncate(r2_score(true_agg, predicted_agg), 2)
	mape = truncate(calcMape(true_agg, predicted_agg), 2) 

	print("R2 = {}, RMSE = {}, MAE = {}, MAPE = {}%".format(r2, rmse, mae, mape))
	return {
		'R2' : r2,
		'RMSE' : rmse,
		'MAE' : mae,
		'MAPE' : "{}%".format(mape)
	}

# https://www.statology.org/mape-python/
def calcMape(actual, pred): 
	actual, pred = np.array(actual), np.array(pred) 
	actual[abs(actual )< 0.1] = 0.1 # A meh hack to avoid division by 0

	return np.mean(np.abs((actual - pred) / actual )) * 100

# https://kodify.net/python/math/truncate-decimals/	
def truncate(number, decimals=0):
    """
    Returns a value truncated to a specific number of decimal places.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor

#=====================================================================
def buildLinearModel(isBinary, label_width, inputCnt):
	_activation, _loss, _metrics = getActivationLossAndMetrics(isBinary)
	model = tf.keras.Sequential([
	
	# Use all time steps
	tf.keras.layers.Flatten(),

	tf.keras.layers.Dense(units=label_width, activation = _activation, kernel_initializer=tf.initializers.zeros()),
	tf.keras.layers.Reshape([label_width, 1]),

	])
	model.compile(loss=_loss, optimizer='adam', metrics = _metrics)

	return model


def buildSimpleNNModel(isBinary, label_width, inputCnt):
	_activation, _loss, _metrics  = getActivationLossAndMetrics(isBinary)

	model = tf.keras.Sequential([
		# Use all time steps
		tf.keras.layers.Flatten(),

		tf.keras.layers.Dense(units=inputCnt, activation='relu'),
		tf.keras.layers.Dense(units=label_width, activation = _activation, kernel_initializer=tf.initializers.zeros()),

		# Add back the time dimension.
		# Shape: (outputs) => (1, outputs)
		tf.keras.layers.Reshape([label_width, 1]),
	])
	model.compile(loss=_loss, optimizer='adam', metrics = _metrics)
	return model

def buildDNNModel(isBinary, label_width, inputCnt):
	_activation, _loss, _metrics = getActivationLossAndMetrics(isBinary)

	model = tf.keras.Sequential([
		 # Use all time steps
		tf.keras.layers.Flatten(),

		tf.keras.layers.Dense(units=inputCnt, activation='relu'),
		tf.keras.layers.Dense(units=inputCnt * 0.80, activation='relu'),

		tf.keras.layers.Dense(units=label_width, activation = _activation, kernel_initializer=tf.initializers.zeros()),

		# Add back the time dimension.
		# Shape: (outputs) => (1, outputs)
		tf.keras.layers.Reshape([label_width, 1]),
	])
	model.compile(loss=_loss, optimizer='adam', metrics = _metrics)
	return model

def buildConvModel(isBinary, lookbackHours, label_width, inputCnt):
	_activation, _loss, _metrics = getActivationLossAndMetrics(isBinary)
	CONV_WIDTH = 4

	model = tf.keras.Sequential([
		# Shape [batch, time, features] => [batch, CONV_WIDTH, features]
		tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),

		tf.keras.layers.Conv1D(filters=inputCnt / 2,
		                       kernel_size=(CONV_WIDTH),
		                       activation='relu'),
		tf.keras.layers.Dense(units=label_width, activation=_activation, kernel_initializer=tf.initializers.zeros()),
		tf.keras.layers.Reshape([label_width, 1]),
		])

	model.compile(loss=_loss, optimizer='adam', metrics = [_metrics])
	return model

def buildLSTMModel(isBinary, label_width, inputCnt):
	_activation, _loss, _metrics = getActivationLossAndMetrics(isBinary)
	model = tf.keras.models.Sequential([
		# Shape [batch, time, features] => [batch, lstm_units]
		tf.keras.layers.LSTM(int(inputCnt / 10), return_sequences=False),

		# Shape => [batch, time, features]
		tf.keras.layers.Dense(units=label_width, activation=_activation, kernel_initializer=tf.initializers.zeros()),
		tf.keras.layers.Reshape([label_width, 1]),
	])

	model.compile(loss=_loss, optimizer='adam', metrics = [_metrics])
	return model

REGRESSION_METRICS = [ tf.keras.metrics.RootMeanSquaredError()]
CLASSIFICATION_METRICS =[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
def getActivationLossAndMetrics(isBinary):
	activation, loss, metrics = "linear", 'mean_squared_error', REGRESSION_METRICS
	if isBinary:
		activation, loss, metrics = "sigmoid", tf.keras.losses.BinaryCrossentropy(), CLASSIFICATION_METRICS

	return activation, loss, metrics

#======================================================================
# Organizing class to encapsulate learning task parameters
class LearningMission():

	def __init__(self, targetLocation, # File containing the data for *where* we are predicting data (e.g. Chicago)
					adjacentLocations, # Files containing the data of nearby locations that are used by the model (e.g. St Louis and Des Moines)
					predictedVariable, # Which variable are we predicting (e.g. WindSpeed)
					featuresToUse,  # What features we are using to predict that variable (e.g. Wind Direction)
					lookAheadHrs,   # How far in advance is the forecast (e.g. 24h)
					lookBackHours,  # How far back are we to look back from the current moment (e.g. 6h)
					aggHalfInterval, # We are not predicting for a single hour but a range of hours : [targetHr - aggHalfInterval .. targetHr + aggHalfInterval]
									# (if this parameter is 0, only predict for the target hour )
					modelToUse # What type of model we'll be using: LINEAR, NN,  DNN, CNN or RNN				
					):  
		self.targetLocation = targetLocation
		self.adjacentLocations  = adjacentLocations
		self.predictedVariable = predictedVariable
		self.featuresToUse = featuresToUse
		self.lookAheadHrs = lookAheadHrs
		self.lookBackHours = lookBackHours
		self.aggHalfInterval = aggHalfInterval
		self.modelToUse = modelToUse

	def __repr__(self):
		return "Predict {} {}h ahead using {} model (locations = {},  features = {}, lookback = {}h, agg interval = {}h)".format(
			self.predictedVariable.upper(), self.lookAheadHrs, self.modelToUse, len(self.adjacentLocations), len(self.featuresToUse), self.lookBackHours, 2 * self.aggHalfInterval + 1)

#======================================================================

class PredictionTarget():
	def __init__(self, var, predictionInterval):
		self.var = var
		self.predictionInterval = predictionInterval

	def __repr__(self):
		return "TARGET {}+{}H".format(self.var, self.predictionInterval)


_is_clear_6h_pt = PredictionTarget('_is_clear', 6)
_is_clear_12h_pt = PredictionTarget('_is_clear', 12)
_is_clear_18h_pt = PredictionTarget('_is_clear', 18)
_is_clear_24h_pt = PredictionTarget('_is_clear', 24)

_is_precip_6h_pt = PredictionTarget('_is_precip', 6)
_is_precip_12h_pt = PredictionTarget('_is_precip', 12)
_is_precip_18h_pt = PredictionTarget('_is_precip', 18)
_is_precip_24h_pt = PredictionTarget('_is_precip', 24)

Temp_6h_pt = PredictionTarget('Temp', 6)
Temp_12h_pt = PredictionTarget('Temp', 12)
Temp_18h_pt = PredictionTarget('Temp', 18)
Temp_24h_pt = PredictionTarget('Temp', 24)

WindSpeed_6h_pt = PredictionTarget('WindSpeed', 6)
WindSpeed_12h_pt = PredictionTarget('WindSpeed', 12)
WindSpeed_18h_pt = PredictionTarget('WindSpeed', 18)
WindSpeed_24h_pt = PredictionTarget('WindSpeed', 24)

prediction_targets = [_is_clear_6h_pt, _is_clear_12h_pt, _is_clear_18h_pt, _is_clear_24h_pt,
					  _is_precip_6h_pt, _is_precip_12h_pt, _is_precip_18h_pt, _is_precip_24h_pt,
					  Temp_6h_pt, Temp_12h_pt, Temp_18h_pt, Temp_24h_pt,
					  WindSpeed_6h_pt,  WindSpeed_12h_pt,  WindSpeed_18h_pt,  WindSpeed_24h_pt]

pt_to_features = {}
pt_to_features[_is_clear_6h_pt] = ['_day_cos', 'DewPoint', '_cloud_intensity', 'CloudAltitude', '_is_thunder']
pt_to_features[_is_clear_12h_pt] = ['_day_cos', 'Temp', 'PressureChange', '_cloud_intensity', 'CloudAltitude', '_wind_dir_sin', '_is_precip']
pt_to_features[_is_clear_18h_pt] =  ['_day_cos', '_hour_cos', 'Temp', 'Pressure', 'PressureChange', 'WindSpeed', '_wind_dir_sin']
pt_to_features[_is_clear_24h_pt] =  ['_day_cos', 'Temp', 'Pressure', 'PressureChange']

pt_to_features[_is_precip_6h_pt] =  ['_hour_sin', 'DewPoint']
pt_to_features[_is_precip_12h_pt] =  ['Precipitation', 'Pressure', 'CloudAltitude', 'WindSpeed', 'WindGust', '_is_clear', '_is_thunder']
pt_to_features[_is_precip_18h_pt] =  ['_day_cos', 'Temp', 'Pressure', 'PressureChange', '_cloud_intensity', 'WindGust', '_wind_dir_sin', '_wind_dir_cos', 'Visibility', '_is_clear', '_is_thunder']
pt_to_features[_is_precip_24h_pt] = ['_day_cos', 'Temp', 'Humidity', 'Pressure', 'PressureChange', '_cloud_intensity', 'WindGust', '_wind_dir_sin', '_wind_dir_cos', 'Visibility', '_is_snow', '_is_thunder']

pt_to_features[Temp_6h_pt] = ['_hour_sin', 'DewPoint', 'Humidity', 'PressureChange', '_cloud_intensity', '_is_thunder']
pt_to_features[Temp_12h_pt] =  ['_hour_sin', 'Precipitation', 'PressureChange', '_cloud_intensity', 'WindGust']
pt_to_features[Temp_18h_pt] = ['_hour_sin', 'Precipitation', 'PressureChange', '_cloud_intensity', 'WindGust', '_wind_dir_sin', '_is_thunder']
pt_to_features[Temp_24h_pt] =  ['DewPoint', 'Precipitation', 'Pressure', '_cloud_intensity', 'WindSpeed', '_wind_dir_cos']

pt_to_features[WindSpeed_6h_pt] = ['_hour_sin', '_hour_cos', 'Humidity', 'Pressure', 'PressureChange']
pt_to_features[WindSpeed_12h_pt] = ['_day_cos', '_hour_sin', '_hour_cos', 'DewPoint', 'Humidity', 'Pressure', 'PressureChange', 'CloudAltitude', 'WindGust', '_wind_dir_sin', '_wind_dir_cos']
pt_to_features[WindSpeed_18h_pt] = ['_day_cos', '_hour_sin', 'Temp', 'PressureChange', '_cloud_intensity', 'CloudAltitude']
pt_to_features[WindSpeed_24h_pt] =  ['_day_cos', '_hour_sin', 'Humidity', 'Precipitation', 'PressureChange', 'WindGust', '_wind_dir_cos', '_is_thunder']

pt_to_locations = {}
pt_to_locations[_is_clear_6h_pt] = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', '../processed-data/noaa_2011-2020_madison_PREPROC.csv', 
									'../processed-data/noaa_2011-2020_st-louis_PREPROC.csv', '../processed-data/noaa_2011-2020_green-bay_PREPROC.csv']
pt_to_locations[_is_clear_12h_pt] = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
									 '../processed-data/noaa_2011-2020_quincy_PREPROC.csv', '../processed-data/noaa_2011-2020_madison_PREPROC.csv', '../processed-data/noaa_2011-2020_st-louis_PREPROC.csv', 
									 '../processed-data/noaa_2011-2020_green-bay_PREPROC.csv']
pt_to_locations[_is_clear_18h_pt] =  ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
									  '../processed-data/noaa_2011-2020_quincy_PREPROC.csv', '../processed-data/noaa_2011-2020_columbus_PREPROC.csv']
pt_to_locations[_is_clear_24h_pt] =  ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
									  '../processed-data/noaa_2011-2020_madison_PREPROC.csv', '../processed-data/noaa_2011-2020_st-louis_PREPROC.csv', '../processed-data/noaa_2011-2020_green-bay_PREPROC.csv', 
									  '../processed-data/noaa_2011-2020_lansing_PREPROC.csv', '../processed-data/noaa_2011-2020_indianapolis_PREPROC.csv']

pt_to_locations[_is_precip_6h_pt] =  ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_quincy_PREPROC.csv', 
									  '../processed-data/noaa_2011-2020_madison_PREPROC.csv', '../processed-data/noaa_2011-2020_green-bay_PREPROC.csv', '../processed-data/noaa_2011-2020_indianapolis_PREPROC.csv']
pt_to_locations[_is_precip_12h_pt] =  ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
									   '../processed-data/noaa_2011-2020_quincy_PREPROC.csv', '../processed-data/noaa_2011-2020_madison_PREPROC.csv', '../processed-data/noaa_2011-2020_st-louis_PREPROC.csv']
pt_to_locations[_is_precip_18h_pt] =  ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
									   '../processed-data/noaa_2011-2020_quincy_PREPROC.csv', '../processed-data/noaa_2011-2020_st-louis_PREPROC.csv', '../processed-data/noaa_2011-2020_green-bay_PREPROC.csv']
pt_to_locations[_is_precip_24h_pt] = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
									 '../processed-data/noaa_2011-2020_madison_PREPROC.csv', '../processed-data/noaa_2011-2020_green-bay_PREPROC.csv', '../processed-data/noaa_2011-2020_lansing_PREPROC.csv', 
									 '../processed-data/noaa_2011-2020_indianapolis_PREPROC.csv']

pt_to_locations[Temp_6h_pt] = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv']
pt_to_locations[Temp_12h_pt] =   ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
								  '../processed-data/noaa_2011-2020_quincy_PREPROC.csv', '../processed-data/noaa_2011-2020_madison_PREPROC.csv', '../processed-data/noaa_2011-2020_lansing_PREPROC.csv', 
								  '../processed-data/noaa_2011-2020_indianapolis_PREPROC.csv']
pt_to_locations[Temp_18h_pt] = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
								'../processed-data/noaa_2011-2020_quincy_PREPROC.csv']
pt_to_locations[Temp_24h_pt] =  ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_st-louis_PREPROC.csv', 
								 '../processed-data/noaa_2011-2020_columbus_PREPROC.csv']

pt_to_locations[WindSpeed_6h_pt] = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_quincy_PREPROC.csv', 
									'../processed-data/noaa_2011-2020_madison_PREPROC.csv', '../processed-data/noaa_2011-2020_st-louis_PREPROC.csv', '../processed-data/noaa_2011-2020_lansing_PREPROC.csv', 
									'../processed-data/noaa_2011-2020_indianapolis_PREPROC.csv']
pt_to_locations[WindSpeed_12h_pt] = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
									 '../processed-data/noaa_2011-2020_quincy_PREPROC.csv', '../processed-data/noaa_2011-2020_indianapolis_PREPROC.csv']
pt_to_locations[WindSpeed_18h_pt] = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_madison_PREPROC.csv', 
									 '../processed-data/noaa_2011-2020_st-louis_PREPROC.csv']
pt_to_locations[WindSpeed_24h_pt] =   ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', 
									 '../processed-data/noaa_2011-2020_quincy_PREPROC.csv']
ahiDict = {'Temp' : 1,
		   'WindSpeed': 2,
		   '_is_precip': 3,
		   '_is_clear': 3,
		   '_is_thunder': 4,
		   '_is_snow' : 3 
		   }

learningMissions = []
for prediction_target in prediction_targets:
	for modelType in ['LINEAR', 'NN', 'DNN', 'CNN']:
		lm = LearningMission(targetLocation = '../processed-data/noaa_2011-2020_chicago_PREPROC.csv',
							 adjacentLocations = pt_to_locations[prediction_target],
							 predictedVariable = prediction_target.var,
							 featuresToUse = pt_to_features[prediction_target],
							 lookAheadHrs = prediction_target.predictionInterval,
							 lookBackHours = 24 if (modelType == 'CNN' or modelType == 'RNN') else 4,
							 aggHalfInterval = ahiDict[prediction_target.var],
							 modelToUse = modelType
	 					    )
		learningMissions.append(lm)

#==========================================================================

summaries = []
for mission in learningMissions:
	evaluation = exploreModel(mission)
	summaries.append(["{}".format(mission), evaluation])

	print("=================\r\n")
	for summary in summaries:
		print(summary[0])
		print(summary[1])
		print("=================\r\n")	

