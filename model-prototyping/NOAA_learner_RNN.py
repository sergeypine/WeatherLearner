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

pd.set_option('display.max_rows', 250)
pd.set_option('display.max_columns', 250)

#=====================================================================
def exploreModel(mission, trainingEpochs = 20):

	# Load the Target Location and Adjacent Locations Datasets and drop the unused columns
	featureset = buildFeatureSet(mission.targetLocation, mission.adjacentLocations, mission.predictedVariable, mission.featuresToUse)

	# Split the data
	column_indices = {name: i for i, name in enumerate(featureset.columns)}
	n = len(featureset)
	train_df = featureset[0 : int(n*0.80)]
	val_df = featureset[int(n*0.80) : ]
	#train_df, val_df = piecewiseSplit(featureset, 8, 2, 3)

	# Normalize input data (NOTE: TF tutorial also scales the target variable)
	train_df, val_df = normalizeData(train_df, val_df, mission.predictedVariable, mission.featuresToUse, len(mission.adjacentLocations))

	agg_interval = 2 * mission.aggHalfInterval + 1
	wg = WindowGenerator(input_width=mission.lookBackHours, 
						 label_width = agg_interval, 
						 shift= mission.lookAheadHrs - mission.aggHalfInterval, 
						 label_columns=[mission.predictedVariable], 
						 train_df = train_df, val_df = val_df, test_df = None)
		

	for example_inputs, example_labels in wg.train.take(1):
		print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
		print(f'Labels shape (batch, time, features): {example_labels.shape}')

	model = None
	if mission.modelToUse == 'LINEAR':
		model = buildLinearModel("is_" in mission.predictedVariable, agg_interval)

	if mission.modelToUse == 'DNN':
		model = buildDNNModel("is_" in mission.predictedVariable, agg_interval)
	
	if mission.modelToUse == 'CNN':
		model = buildConvModel("is_" in mission.predictedVariable, mission.lookBackHours, agg_interval)
	
	if mission.modelToUse == 'RNN':
		model = buildLSTMModel("is_" in mission.predictedVariable, agg_interval)

	model.fit(wg.train, epochs = trainingEpochs)	

	eval_func = evaluateClassificationModel if "is_" in mission.predictedVariable else evaluateRegressionModel

	print("- Performance on *TRAINING* data ({}):".format(mission))
	eval_func(model, wg.train)
	print("- Performance on *TEST* data ({}):".format(mission))
	return eval_func(model, wg.val)

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
def dropUnusedColumns(df, predictedVariable, featuresToUse):
	all_columns = featuresToUse.copy()
	all_columns.append('DATE')
	all_columns.append(predictedVariable)
	df = df[all_columns]

	return df

#======================================================================
def normalizeData(trainDf, valDf,  predictedVariable, featuresToUse, adjacentLocationCount):

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

#======================================================================		
#======================================================================
# 	BEGIN Data reformatting, taken from https://www.tensorflow.org/tutorials/structured_data/time_series
#======================================================================
#======================================================================
class WindowGenerator():

	def __init__(self, input_width, label_width, shift,
	           train_df, val_df, test_df,
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

def evaluateClassificationModel(model, testSet):

	predicted_labels =(model.predict(testSet, verbose = 1) > 0.5).astype("int32")
	#predicted_labels =  np.concatenate([y for y in predicted_labels], axis=0)
	true_labels = np.concatenate([y for x, y in testSet], axis=0)

	assert len(predicted_labels) == len (true_labels)

	# We are forecasting for a number of hours: aggregate each forecast series using the "True iff 1 or more is True" rule
	predicted_agg = []
	true_agg = []
	for i in range(0, len(predicted_labels)):
		predicted_i = predicted_labels[i].flatten()
		true_i = true_labels[i].flatten()

		predicted_i_agg = 1 if sum(predicted_i) > 0 else 0
		true_i_agg = 1 if sum(true_i) > 0 else 0

		#print("{} => {}".format(predicted_i, predicted_i_agg))
		#print("{} => {}".format(true_i, true_i_agg))
		#print("----")

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

def evaluateRegressionModel(model, testSet):
	predicted_values = model.predict(testSet)
	#predicted_values = np.concatenate([y for y in predicted_values], axis=0)
	true_values = np.concatenate([y for x, y in testSet], axis=0)

	assert len(predicted_values) == len (true_values)

	predicted_agg = []
	true_agg = []
	for i in range(0, len(predicted_values)):
		predicted_i = predicted_values[i].flatten()
		true_i = true_values[i].flatten()

		predicted_i_agg, true_i_agg = np.mean(predicted_i), np.mean(true_i)
		#print("True: {} => {}".format(true_i, true_i_agg))
		#print("Predicted: {} => {}".format(predicted_i, predicted_i_agg))
		#print("-------\r\n\r\n")

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
    actual[actual == 0] = 0.1 # A meh hack to avoid division by 0

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
def buildLinearModel(isBinary, label_width):
	_activation, _loss = getActivationAndLoss(isBinary)
	model = tf.keras.Sequential([
	 	# Take the last time-step.
    	# Shape [batch, time, features] => [batch, 1, features]
    	tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),

	    tf.keras.layers.Dense(units=label_width, activation = _activation, kernel_initializer=tf.initializers.zeros()),
	    tf.keras.layers.Reshape([label_width, 1]),
	])
	model.compile(loss=_loss, optimizer='adam', metrics = [keras.metrics.BinaryAccuracy(name='accuracy')])
	return model


def buildDNNModel(isBinary, label_width):
	_activation, _loss = getActivationAndLoss(isBinary)

	model = tf.keras.Sequential([
     # Take the last time step.
     # Shape [batch, time, features] => [batch, 1, features]
     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
     tf.keras.layers.Dense(units=200, activation='relu'),
     tf.keras.layers.Dense(units=200, activation='relu'),
     tf.keras.layers.Dense(units=label_width, activation = _activation, kernel_initializer=tf.initializers.zeros()),
     # Add back the time dimension.
     # Shape: (outputs) => (1, outputs)
     tf.keras.layers.Reshape([label_width, 1]),
	])
	model.compile(loss=_loss, optimizer='adam', metrics = [keras.metrics.BinaryAccuracy(name='accuracy')])
	return model

def buildConvModel(isBinary, lookbackHours, label_width):
	_activation, _loss = getActivationAndLoss(isBinary)
	CONV_WIDTH = 4

	model = tf.keras.Sequential([
		# Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    	tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),

	    tf.keras.layers.Conv1D(filters=200,
	                           kernel_size=(CONV_WIDTH),
	                           activation='relu'),
	    tf.keras.layers.Dense(units=200, activation='relu'),
	    tf.keras.layers.Dense(units=label_width, activation=_activation, kernel_initializer=tf.initializers.zeros()),
	    tf.keras.layers.Reshape([label_width, 1]),
	])

	model.compile(loss=_loss, optimizer='adam', metrics = ['accuracy'])
	return model

def buildLSTMModel(isBinary, label_width):
	_activation, _loss = getActivationAndLoss(isBinary)
	model = tf.keras.models.Sequential([

		# Shape [batch, time, features] => [batch, lstm_units]
    	tf.keras.layers.LSTM(32, return_sequences=True),
    	tf.keras.layers.LSTM(32, return_sequences=False),
    	# TODO - understand this a little better
    	
    	# Shape => [batch, time, features]
    	tf.keras.layers.Dense(units=label_width, activation=_activation, kernel_initializer=tf.initializers.zeros()),
    	tf.keras.layers.Reshape([label_width, 1]),
	])


	model.compile(loss=_loss, optimizer='adam', metrics = ['accuracy'])
	return model

def getActivationAndLoss(isBinary):
	activation, loss = "linear", 'mean_absolute_error'
	if isBinary:
		activation, loss = "sigmoid", "binary_crossentropy"
	return activation, loss

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
					modelToUse # What type of model we'll be using: LINEAR, DNN, CNN or RNN				
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
		return "Predict {} {}h ahead using {} model (location count = {},  feature count = {}, lookback = {}h, agg interval = {}h)".format(
			self.predictedVariable.upper(), self.lookAheadHrs, self.modelToUse, len(self.adjacentLocations), len(self.featuresToUse), self.lookBackHours, 2 * self.aggHalfInterval + 1)

#======================================================================

featureDict = {'Temp' : ['_day_sin', '_day_cos', '_hour_sin', '_hour_cos', '_cloud_intensity', 'DewPoint', 'WindSpeed', '_wind_dir_sin', '_wind_dir_cos'],
			   'WindSpeed': ['_day_sin', '_day_cos', '_hour_sin', '_hour_cos', '_wind_dir_sin', '_wind_dir_cos', '_cloud_intensity', 'Temp', 'CloudAltitude', 'Pressure'],
			   '_is_precip': ['_cloud_intensity', 'CloudAltitude', 'Temp', 'Precipitation', '_is_snow', '_is_thunder', 'Humidity',  'WindSpeed', '_wind_dir_sin', '_wind_dir_cos'],
			   '_is_cloudy': ['_cloud_intensity', 'CloudAltitude', 'Temp', 'Precipitation', '_is_mist', 'Humidity', 'WindSpeed', '_wind_dir_sin', '_wind_dir_cos']}

learningMissions = []
for var in ['_is_precip']:
	for predictionHrs in [12]:
		for modelType in ['DNN']:
			lm = LearningMission(targetLocation = '../processed-data/noaa_2011-2020_chicago_PREPROC.csv',
								 adjacentLocations = ['../processed-data/noaa_2011-2020_cedar-rapids_PREPROC.csv', '../processed-data/noaa_2011-2020_des-moines_PREPROC.csv', 
								 					  '../processed-data/noaa_2011-2020_rochester_PREPROC.csv', '../processed-data/noaa_2011-2020_quincy_PREPROC.csv',
								 					  '../processed-data/noaa_2011-2020_madison_PREPROC.csv', '../processed-data/noaa_2011-2020_st-louis_PREPROC.csv',
								 					  '../processed-data/noaa_2011-2020_green-bay_PREPROC.csv', '../processed-data/noaa_2011-2020_lansing_PREPROC.csv'],
								 predictedVariable = var,
								 featuresToUse = featureDict[var],
								 lookAheadHrs = predictionHrs,
								 lookBackHours = 12,
								 aggHalfInterval = 3 if '_is' in var else 1,
								 modelToUse = modelType
		 					    )
			learningMissions.append(lm)

#==========================================================================

summaries = []
for mission in learningMissions:
	evaluation = exploreModel(mission, 30)
	summaries.append(["{}".format(mission), evaluation])

print("=================\r\n")
for summary in summaries:
	print(summary[0])
	print(summary[1])
	print("=================\r\n")

