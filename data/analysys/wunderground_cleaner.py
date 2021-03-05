import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 250)

#=================================================

def cleanFile(filePath):
	cleaned_df = None
	raw_df = pd.read_csv(filePath, parse_dates = ['Time'])

	#Drop "F" in temperatures
	raw_df['Temp'] = raw_df['Temp'].str.replace(' F', '')
	raw_df['DewPoint'] = raw_df['DewPoint'].str.replace(' F', '')

	#Drop % in Humidty
	raw_df['Humidity'] = raw_df['Humidity'].str.replace(' %', '')

	#Drop mph in Windspeed and WindGust
	raw_df['WindSpeed'] = raw_df['WindSpeed'].str.replace(' mph', '')
	raw_df['WindGust'] = raw_df['WindGust'].str.replace(' mph', '')

	#Drop in from Pressure
	raw_df['Pressure'] = raw_df['Pressure'].str.replace(' in', '')
	
	#Drop Precipitation column-- it appears to be wrongly populated
	raw_df = raw_df.drop(columns = ['Precip'])

	#Transform wind direction into 3 components: Northerly, Easterly and is_var
	raw_df['WindNortherly'], raw_df['WindEasterly'], raw_df['is_var'] = zip(*raw_df['Wind'].apply(getWindDirectionNEComponents))

	#Transform condition into 7 flags: is_clear, is_fog, is_precip, is_tstorm, cloud_intensity, precip_intensity
	raw_df['is_clear'], raw_df['is_fog'], raw_df['is_precip'], raw_df['is_tstorm'], raw_df['cloud_intensity'], raw_df['precip_intensity'] = zip(
		*raw_df['Condition'].apply(convertConditionToFlags))

	#************
	cleaned_df = raw_df
	#*************

	#Round all timestamps to 1h
	cleaned_df['Time'] = cleaned_df['Time'].dt.round('1h')

	#Cutoff all data before 03/15/2014 - it is missing precipitation data
	cleaned_df = cleaned_df[cleaned_df['Time'] > pd.to_datetime('2014-03-14')]
	cleaned_df = cleaned_df.sort_values(by ='Time')

	#Drop rows with identical timestamp (for some hours we have multiple readings, keep one)
	#NOTE - this chooses a random duplicate value, consider doing an average instead
	cleaned_df = cleaned_df.drop_duplicates(subset = ['Time']) 

	#Fill in missing hours and impune values but only within 6h
	cleaned_df = cleaned_df.set_index('Time').resample('H').nearest(limit = 6).dropna().reset_index()


	# Create day-of-year and hour-of-day sin/cos columns to capture the cyclical nature of time

	cleaned_df['day_of_year_sin'] = np.sin(2 * np.pi * cleaned_df['Time'].dt.dayofyear / 365)
	cleaned_df['day_of_year_cos'] = np.cos(2 * np.pi * cleaned_df['Time'].dt.dayofyear / 365)

	cleaned_df['hour_of_day_sin'] = np.sin(2 * np.pi * cleaned_df['Time'].dt.hour / 24)
	cleaned_df['hour_of_day_cos'] = np.cos(2 * np.pi * cleaned_df['Time'].dt.hour / 24)

	return cleaned_df

#=================================================
wind_NE_component_dict = {
	'W': [0, -1, 0],
	'WSW':  [-0.25, -0.75, 0],
	'SSW': [-0.75, -0.25, 0] ,
	'SW': [-0.5, -0.5, 0], 
	'S': [-1, 0, 0], 
	'WNW' : [0.25, -0.75, 0],
	'NW': [0.5, -0.5, 0], 
	'CALM': [0, 0, 0], 
	'NNW': [0.75, -0.25, 0], 
	'N': [1, 0, 0],
	'VAR' : [0, 0, 1],
	'ENE' : [0.25, 0.75, 0],
	'NNE': [0.75, 0.25, 0], 
	'ESE': [-0.25, 0.75, 0],
	'SSE': [-0.75, 0.25, 0],
	'SE': [-0.5, 0.5, 0], 
	'E': [0, 1, 0], 
	'NE': [0.5, 0.5, 0]
}
#=================================================
def getWindDirectionNEComponents(windDirection):
	return wind_NE_component_dict[windDirection][0],wind_NE_component_dict[windDirection][1],wind_NE_component_dict[windDirection][2]

#=================================================
def convertConditionToFlags(condition):
	is_clear = 0
	is_fog = 0
	is_precip = 0
	is_tstorm = 0
	
	precip_intensity = 0
	cloud_intensity = 3
	# 0: not cloudy 
	# 1: partly cloudy
	# 2: mostly cloudy
	# 3: cloudy

	lcondition = condition.lower()


	if (('fair' in lcondition) or ('smoke' in lcondition) or ('haze' in lcondition)):
		is_clear = 1
		cloud_intensity = 0

	if ('partly' in lcondition):
		is_clear = 1
		cloud_intensity = 1

	if ('mostly' in lcondition):
		cloud_intensity = 2

	if 'fog' in lcondition:
		is_fog = 1
		cloud_intensity = 1 # looked at data, seems like having 1 here makes it smoother		

	if (('rain' in lcondition) or ('drizzle' in lcondition) or ('snow' in lcondition) or ('t-storm' in lcondition) or ('wintry mix' in lcondition) or ('hail' in lcondition) or ('precipitation' in lcondition)):
		precip_intensity = 2
		cloud_intensity = 3
		is_precip = 1

	if 'heavy' in lcondition:
		precip_intensity = 3
		is_heavy_precip = 1	

	if 'light' in lcondition:
		precip_intensity = 1
		is_light_precip = 1

	if (('t-storm' in lcondition) or ('thunder' in lcondition)):
		is_tstorm = 1

	return is_clear, is_fog, is_precip, is_tstorm, cloud_intensity, precip_intensity

#=================================================
def getCleanFileName(filePath):
	return filePath.replace('.csv', '_CLEANED.csv')

#=================================================

files_to_clean = ['../Chicago_2011-2020.csv', 
	'../Toledo_2011-2020.csv', '../StLouis_2011-2020.csv', '../Peoria_2011-2020.csv', '../CedarRapids_2011-2020.csv', '../Saginaw_2011-2020.csv', '../FortWayne_2011-2020.csv', '../Milwaukee_2011-2020.csv',
	'../Indianapolis_2011-2020.csv', '../Madison_2011-2020.csv', '../GreenBay_2011-2020.csv', '../GrandRapids_2011-2020.csv',
	'../Cleveland_2011-2020.csv', '../Columbus_2011-2020.csv', '../Cincinatti_2011-2020.csv', '../Des_Moines_2011-2020.csv', '../Minneapolis_2011-2020.csv', 
	'../Sault_Ste_Marie_2011-2020.csv', '../Pittsburgh_2011-2020.csv', '../Toronto_2011-2020.csv', '../KansasCity_2011-2020.csv', '../Duluth_2011-2020.csv']
cleaned_count = 0
for file_to_clean in files_to_clean:
	cleaned_df = cleanFile(file_to_clean)
	cleaned_df.to_csv(getCleanFileName(file_to_clean), index = False)
	cleaned_count = cleaned_count + 1
	print("Cleaned & Saved {} files out of {}, line count = {}".format(cleaned_count, len(files_to_clean), len(cleaned_df)))
