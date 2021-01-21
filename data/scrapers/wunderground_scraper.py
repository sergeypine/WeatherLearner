import time
import datetime
import random
import itertools
from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import sys


#=====================================================================================
PAGE_LOAD_RETRIES = 10


#=====================================================================================
#URL example: https://www.wunderground.com/history/daily/KMDW/date/2015-1-21

_driver = webdriver.Chrome(executable_path="/home/snumerov/ucsd/WeatherLearner/data/scrapers/chromedriver")
def getHtmlForDate(date, stationCode):
	target_url = 'https://www.wunderground.com/history/daily/{}/date/{}'.format(stationCode, date)
	

	attempts = 0

	while attempts < PAGE_LOAD_RETRIES:
		try:
			_driver.get(target_url)

			soup = BeautifulSoup(_driver.page_source, 'lxml')
			tableHtml = soup.find_all('table')[1]
			return tableHtml
		except:
			attempts = attempts + 1
			if attempts >= PAGE_LOAD_RETRIES:
				print("Unexpected error loading page:", sys.exc_info()[0], sys.exc_info()[1])

	return None

#=====================================================================================
#Time, Temperature, Dew Point, Humidity, Wind, Wind Speed, Wind Gust,  Pressure, Precip.,      Condition,
def getHourlyDataForDateFromHtml(pageHtml, date):

	hourly_data_list = []
	
	try:
		df = pd.read_html(str(pageHtml))[0]

		#get rid of NAN's
		df = df.dropna()

		#prepend date  to time
		df['Time'] = df['Time'].apply(lambda x: "{} {}".format(date, x))

		# Iterate over each row
		for index, row in df.iterrows():
			hourly_data_list.append([
				row['Time'], 
				row['Temperature'],
				row['Dew Point'],
				row['Humidity'],
				row['Wind'],
				row['Wind Speed'],
				row['Wind Gust'],
				row['Pressure'],
				row['Precip.'],
				row['Condition']])


		print(hourly_data_list)

	except:
		print("Unexpected error parsing page HTML:", sys.exc_info()[0], sys.exc_info()[1])
		return []

	return hourly_data_list

#=====================================================================================
def saveHourlyDataCsv(hourlyData, csvFile):
	pass

#=====================================================================================
def getNextDayDate(todayDate):
	today_as_datetime = datetime.datetime.strptime(todayDate,'%Y-%m-%d')
	tomorrow_as_datetime = today_as_datetime + datetime.timedelta(days=1)
	return tomorrow_as_datetime.strftime('%Y-%m-%d')

#=====================================================================================
def mainFunc(startYear, yearCount, stationCode, locationName):
	present_date = "{}-{}-{}".format(startYear, '01', '01')
	end_date = "{}-{}-{}".format(startYear + yearCount, '01', '01')
	all_hourly_data = []

	dates_success = 0
	dates_failure = 0
	start_time = time.time()


	while (present_date != end_date):
		html = getHtmlForDate(present_date, stationCode)
		data = []

		if html is not None:
			data = getHourlyDataForDateFromHtml(html, present_date)
			if len(data) == 0:
				dates_failure = dates_failure + 1
			else:
				dates_success = dates_success + 1
				all_hourly_data.extend(data) #don't forget to unnest
		else:
			dates_failure = dates_failure + 1	


		print("Processing date {}; total execution time = {} s, date records = {}, dates success = {}, dates failure = {}".format( 
			present_date, int(time.time() - start_time), len(data), dates_success, dates_failure))
		present_date = getNextDayDate(present_date)


	output_file_name = "{}_{}-{}.csv".format(locationName, startYear, startYear + yearCount - 1)
	saveHourlyDataCsv(all_hourly_data, output_file_name)
	print("Successfully saved {} records to file {}".format(len(all_hourly_data), output_file_name))

#=====================================================================================

mainFunc(2010, 3, 'KDSM', 'Des_Moines')

