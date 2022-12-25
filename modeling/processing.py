import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import time


# import nltk
#
# nltk.download('omw-1.4')
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.stem.wordnet import WordNetLemmatizer


# def getStopWordInEnglish():
#     stop_dict = dict()
#     for w in stopwords.words('english'):
#         stop_dict[w] = True
#     print(list(stop_dict.keys()))
#     return stop_dict
#
#
# def remove_stop_words(string, stop_dict):
#     words = string.split(" ")
#     final_word = [word for word in words if stop_dict.get(word) is None and word != '']
#     return " ".join(final_word)
#
#
# def stemming(string):
#     porter = PorterStemmer()
#     words = string.split(" ")
#     final_words = [porter.stem(word) for word in words]
#     return " ".join(final_words)
#
#
# def lemmatization(string):
#     lemmatizer = WordNetLemmatizer()
#     words = string.split(" ")
#     final_words = [lemmatizer.lemmatize(word) for word in words]
#     return " ".join(final_words)


def drawBoxPlot(data, x_feature_name, path, title=None, y_feature_name=None):
    plt.subplots(figsize=(15, 6))
    sns.set_theme()
    if y_feature_name is None:
        sns.boxplot(data=data, x=x_feature_name).set_title(title)
    else:
        sns.boxplot(data=data, x=x_feature_name, y=y_feature_name).set_title(title)
    plt.savefig(path)
    plt.close()


def drawMeanPlot(data, group_name, feature_name, path):
    plt.subplots(figsize=(15, 6))
    data.groupby(group_name)[feature_name].mean().plot()
    plt.savefig(path)
    plt.close()


def drawCatPlot(data, x_feature_name, y_feature_name, isNormalize, hue_name, plot_kind, title, path):
    sns.set_theme()
    plt.subplots(figsize=(20, 6))
    graph = (data.groupby(x_feature_name)[hue_name]
             .value_counts(normalize=isNormalize)
             .mul(100)
             .rename(y_feature_name)
             .reset_index()
             .pipe((sns.catplot, 'data'), x=x_feature_name, y=y_feature_name, hue=hue_name, kind=plot_kind, aspect=2))
    graph.fig.suptitle(title)
    plt.savefig(path)
    plt.close()


def drawCountPlot(data, x_feature_name, path, title=None, order=None, x_labels=False, labels=None):
    sns.set_theme()
    plt.subplots(figsize=(20, 6))
    graph = sns.countplot(x=x_feature_name, data=data, order=order)
    if title is not None:
        graph.set_title(title)
    if x_labels:
        if labels is None:
            graph.set_xticklabels(graph.get_xticklabels(), rotation=90)
        else:
            graph.set_xticklabels(labels, rotation=90)
    plt.savefig(path)
    plt.close()


def printMissingValues(nan_values, size):
    """
    count null values for each feature and sort them,
    filter out features where there are no missing values
    finding the percentage of missing values
    """
    nan_counts = nan_values.to_frame()
    nan_counts.columns = ['missing_values_count']
    nan_counts = nan_counts.loc[nan_counts['missing_values_count'] > 0]
    nan_percent = nan_values[nan_values != 0] / size * 100
    nan_counts.insert(1, "missing_values_percent", nan_percent)
    return nan_counts


def preProcess(dataSource):
    print("Reading data into pandas dataframe!!")
    # data = pd.read_csv(dataSource)
    # data.to_pickle("D:\projects\CSE587_DIC_Project\data/initial_data.pkl")

    data = pd.read_pickle('D:\projects\CSE587_DIC_Project\data\initial_data.pkl')

    print('Number of data points : ', data.shape[0])
    print('Number of features : ', data.shape[1])
    print('Features List: ', data.columns.values, '\n')
    print('Feature name and datatype for values of each feature: ')
    print(data.info(), '\n')
    print('Feature name and boolean representing if null values are present or not')
    print(data.isnull().any(), "\n")
    print('Duplicated rows: ')
    print(data.duplicated().any(), '\n')

    nan_values = data.isnull().sum().sort_values(ascending=False)
    print("Feature, Total Missing Values, Missing Values Percentage: ")
    print(printMissingValues(nan_values, len(data)), "\n")

    data.drop('Number', inplace=True, axis=1)
    data.drop('ID', inplace=True, axis=1)
    data.drop('Country', inplace=True, axis=1)

    data.dropna(subset=['Wind_Speed(mph)', 'Wind_Direction', 'Humidity(%)', 'Weather_Condition',
                        'Visibility(mi)', 'Temperature(F)', 'Pressure(in)', 'Weather_Timestamp',
                        'Airport_Code', 'Timezone', 'Nautical_Twilight', 'Civil_Twilight', 'Sunrise_Sunset',
                        'Astronomical_Twilight', 'Zipcode', 'City', 'Street'], inplace=True)
    print("Shape of data after preliminary cleaning:", data.shape, "\n")

    nan_values = data.isnull().sum().sort_values(ascending=False)
    print("Feature, Total Missing Values, Missing Values Percentage: --after processing")
    print(printMissingValues(nan_values, len(data)), "\n")

    '''
    * Now, let's look at the **Precipitation(in)** column.
    * Shows precipitation amount in inches, if there is any.
    * About **15.6%** of the data is missing.
    * And of the remaining data, **77.5%** data has precipitation as 0 inches.
    * From this information and results above it is evident that 1st quantile, 2nd quantile and 3rd quantile values are 0.
    * Our mode of data is found out to be **0.000**
    * We will replace our NaN's with the calculated **mode** (most frequent value).
    '''
    data['Precipitation(in)'].fillna(0.000, inplace=True)

    '''
    * Now let's look at **Wind_Chill(F)** column.
    * Shows the wind chill (in Fahrenheit).
    * About **11.3%** of the data is missing.
    * Our 50th percentile value (median) is said to be: 63.0
    * We can replace our NaN's with the median
    '''
    data['Wind_Chill(F)'].fillna(63.0, inplace=True)

    print("Number of columns having NaN's in their records: ",
          len([col for col in data.columns if data[col].isnull().any()]), "\n")

    '''
    * So by this stage all of the missing values in our dataset have been processed
    '''
    print("Wind directions: ", data['Wind_Direction'].value_counts().keys().tolist(), "\n")
    drawCountPlot(data, "Wind_Direction", '../plots/wind_dir_before.png', 'Wind direction of accidents (Before)')

    """
    * "Calm" is the most common wind direction in our data.
    * After analyzing we got to know that some entries have same meaning but are described using different notations. 
    * Below are the examples:
    1. 'S' and 'South'  > both represent the wind coming from the south and blowing toward the north.
    2. 'N' and 'North' > both represent the wind coming from the north and blowing toward the south.
    3. 'E' and 'East' >  both represent the wind coming from the east and blowing toward the west.
    4. 'W' and 'West' >  both represent the wind coming from the west and blowing toward the east.
    5. 'VAR' and 'Variable' > both represent the wind whose direction fluctuates by 60° or more during the 2-minute evaluation period and the wind speed is greater than 6 knots (or) the direction is variable and the wind speed is less than 6 knots.
    * So I have replaced the entires showing 'South' to 'S', 'North' to 'N', 'East' to 'E', 'West' to 'W', 'VAR' to 'Variable'.
    """

    data['Wind_Direction'].replace('South', 'S', inplace=True)
    data['Wind_Direction'].replace('North', 'N', inplace=True)
    data['Wind_Direction'].replace('West', 'W', inplace=True)
    data['Wind_Direction'].replace('East', 'E', inplace=True)
    data['Wind_Direction'].replace('VAR', 'Variable', inplace=True)
    print("Wind directions after replacing: ", data['Wind_Direction'].value_counts().keys().tolist(), "\n")

    drawCountPlot(data, "Wind_Direction", '../plots/wind_dir_after.png', 'Wind direction of accidents (After)')

    """
    * Start-Time: Shows start time of the accident in local time zone.
    * End-Time: Shows end time of the accident in local time zone. End time here refers to when the impact of accident on traffic flow was dismissed.
    * Records of both the columns are of type object.
    * So we will convert it to readable pandas datetime format.
    """

    data['Start_Time'] = pd.to_datetime(data.Start_Time)
    data['End_Time'] = pd.to_datetime(data.End_Time)

    """
    * Now, both "Start_Time" and "End_Time" are in readable format.
    * Inorder to analyze them, we will be breaking these into year, month, day, weekday, hour 
    and will be adding these as new columns to our data-frame.
    """

    data['Start_Year'] = data['Start_Time'].dt.year
    data['Start_Month'] = data['Start_Time'].dt.month
    data['Start_Day'] = data['Start_Time'].dt.day
    data['Start_Weekday'] = data['Start_Time'].dt.weekday
    data['Start_Weekday'] = data['Start_Time'].dt.weekday
    data['Start_Hour'] = data['Start_Time'].dt.hour

    """
    * "Start_Weekday": It is the day of the week returned as a value where Monday=0 and Sunday=6
    * "Start_Hour": Will be in 24 hour format, 12Am is 0 and 11Pm is 23.
    """

    """
    * **Description**: Shows natural language description of the accident.
    * As natural language data doesn't often follow any specific format.
    * Standardizing this data will be good for analysis.
    """

    # data['Cleaned_Description'] = data['Description'].str.lower()  # Converts all the text into lower case.
    # data['Cleaned_Description'] = data[
    #     'Cleaned_Description'].str.strip()  # Helps in removing leading and trailing white spaces.
    # data['Cleaned_Description'] = data['Cleaned_Description'].str.replace('\\.', '',
    #                                                                       regex=True)  # Helps in removing period.
    # data['Cleaned_Description'] = data['Cleaned_Description'].str.replace('\\-', '',
    #                                                                       regex=True)  # Helps in removing '-'.
    # data['Cleaned_Description'] = data['Cleaned_Description'].str.replace('\\/', ' ',
    #                                                                       regex=True)  # Helps in removing '/'.
    # data['Cleaned_Description'] = data['Cleaned_Description'].str.replace('  ', ' ', regex=True)

    """
    * Now we have processed the description up a bit.
    * Let's remove stopwords and apply stemming, lemmatization to the processed text.
    """

    # desc_strings = []
    # for desc in data['Cleaned_Description'].values:
    #     desc_strings.append(lemmatization(stemming(remove_stop_words(desc, getStopWordInEnglish()))))
    # data['Cleaned_Description'] = desc_strings

    """Temperature(F): Shows the temperature (in Fahrenheit)."""

    drawBoxPlot(data, 'Temperature(F)', "../plots/temperature_boxplot_before.png", "Temperature Distribution Before")
    print("Temperature description Before: ")
    print(data['Temperature(F)'].describe(), "\n")

    print("records having 'Temperature(F)' > 100 =",
          len(data.loc[(data['Temperature(F)'] > 100)]))  # filters records having 'Temperature(F)' > 100
    print("records having 'Temperature(F)' < -20 =",
          len(data.loc[(data['Temperature(F)'] < -20)]))  # filters records having 'Temperature(F)' < -20

    """
    * Based on a wikipedia blog (https://en.wikipedia.org/wiki/U.S._state_and_territory_temperature_extremes), highest temperature ever recorded in United states is about 134 °F in 1913.
    * And in the recent times (which is in between 2016 and 2021), highest temperature recorded is 120 °F.
    * Also there are very very few observations having temperature higher than 100 °F and lower than -20 °F, but in our dataset we have about 9450 observations (about **0.37%** of our data) with these constraints. 
    * So we will be treating these as outliers and replace them with the **mean** temperature of the respective states.
    """
    for state in data['State'].unique().tolist():
        tempList = data.loc[(data['Temperature(F)'] <= 100) & (data['Temperature(F)'] >= -20) &
                            (data['State'] == state)]['Temperature(F)'].tolist()
        mean = sum(tempList) / len(tempList)
        data['Temperature(F)'] = np.where(((data['Temperature(F)'] > 100) |
                                           (data['Temperature(F)'] < -20)) &
                                          (data['State'] == state), mean, data['Temperature(F)'])

    drawBoxPlot(data, 'Temperature(F)', "../plots/temperature_boxplot_after.png", "Temperature Distribution After")

    print("Temperature description After removing possible outliers: ")
    print(data['Temperature(F)'].describe(), "\n")

    """
    * Pressure(in): Shows the air pressure (in inches).
    """
    drawBoxPlot(data, 'Pressure(in)', "../plots/pressure_boxplot_before.png", "Pressure Distribution Before")
    print("Pressure description Before: ")
    print(data['Pressure(in)'].describe(), "\n")

    print("records having 'Pressure(in)' > 32 =",
          len(data.loc[(data['Pressure(in)'] > 32)]))  # filters records having 'Pressure(in)' > 32
    print("records having 'Pressure(in)' < 20 =",
          len(data.loc[(data['Pressure(in)'] < 20)]))  # filters records having 'Pressure(in)' < 20

    """
    * https://wgntv.com/weather/weather-blog/ask-tom-why/what-is-the-highest-air-pressure-value-ever-recorded/#:~:text=Northway%2C%20Alaska%20takes%20the%20honors,record%20cold%20to%20that%20state.
    * https://sciencing.com/high-low-reading-barometric-pressure-5814364.html
    * From the above blogs it's been observed that highest pressure ever observed in US is about 31.85 inches.
    * After analyzing the data in 'Pressure(in)' column, there are very very few records with pressure value greater than 32 and lower than 20, which are about 150 records (constitutes to 0.00006%) of our data.
    * So we will be replacing those values with the median.
    """

    median = data.loc[(data['Pressure(in)'] <= 32) | (data['Pressure(in)'] >= 20)]['Pressure(in)'].median()
    data['Pressure(in)'] = np.where(((data['Pressure(in)'] > 32) | (data['Pressure(in)'] < 20)),
                                    median, data['Pressure(in)'])

    drawBoxPlot(data, 'Pressure(in)', "../plots/pressure_boxplot_after.png", "Pressure Distribution After")
    print("Pressure description After removing possible outliers: ")
    print(data['Pressure(in)'].describe(), "\n")

    """
    * Adding a new column "Duration" to our dataframe.
    * This represents duration of traffic disruption in minutes occurred due to the accident.
    """

    data['Duration(min)'] = (data['End_Time'] - data['Start_Time']).astype('timedelta64[m]')
    drawBoxPlot(data, 'Severity', "../plots/dur_vs_severity_before.png",
                'Distribution of Duration(min) by Severity (Before)', 'Duration(min)')

    """
    * We can observe that distribution of data is not clear at all from the above plot. Because of some rare and very large values.
    * And also there is no significant difference in distribution for various severity levels.
    * So lets filter out some of the records.
    """

    df = data[data['Duration(min)'] <= 800]
    print("Percentage of records having Duration greater than 800 are: ", ((len(data) - len(df)) / len(data)) * 100)
    del df

    """
    * Filtering records with Duration less than 800 minutes (about 12 hours), and the remaining constituted to about **1.92%** of the data.
    * Now let's draw the plot again with the filtered data.
    """
    drawBoxPlot(data[data['Duration(min)'] <= 800], 'Severity', "../plots/dur_vs_severity_after.png",
                'Distribution of Duration(min) by Severity (After)', 'Duration(min)')

    statesList = data.State.unique()
    print("States: ", statesList)
    print("Number of states in Dataset: ", len(statesList))
    print("There are 50 states in US and out of those 49 are in our dataset. \n New York is not in our dataset.", "\n")

    drawCountPlot(data, "State", "../plots/accidents_count_per_state.png",
                  'Accidents Count by State', data.State.value_counts().index)

    """
    * After initial observation it's been observed that california has recorded greater number of accidents than other states.
    * Florida records for the next greater number of accidents and followed by texas.
    * https://www.infoplease.com/us/states/state-population-by-rank
    * California is the most populated state in US, followed by Texas and Florida.
    * From this, we can expect to see a larger number of accidents in states with high population than states with lower ones.
    """

    cityList = data.City.unique()
    print("Cities: ", cityList)
    print("Total No. of Cities in our data: ", len(cityList), "\n")

    """
    * There are 10788 cities in our data.
    * https://worldpopulationreview.com/us-city-rankings/how-many-cities-are-in-the-us
    * As per above blog there are about 19,495 incorporated cities in US as of 2018.
    """
    drawCountPlot(data, "City", "../plots/accidents_count_per_city.png",
                  "Top 50 Cities with Highest No. of Accidents", data.City.value_counts().iloc[:50].index, True)

    drawCountPlot(data, "Start_Weekday", "../plots/accidents_count_per_weekday.png",
                  'Number of Accidents by Day of the Week', None, True,
                  ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    """
    * From the above plot we can observe that number of accidents remain almost same during weekdays (Monday to Friday).
    * Where as they drop during the weekends.
    * We can assume that this could be due to work commute which happens during weekdays.
    """

    drawCatPlot(data, 'Start_Weekday', 'Percent', True, 'Severity', "bar",
                'Severity of Accidents by Day', "../plots/weekday_severity_bar_plot.png")

    """Severity of the accidents remain almost consistent on all the days."""

    plt.subplots(figsize=(15, 6))
    sns.countplot(x="Start_Hour", data=data, orient='v')
    plt.annotate('Peak Hours morning', xy=(6, 210995), fontsize=12)
    plt.annotate('Peak Hours Afternoon', xy=(14, 210995), fontsize=12)
    plt.annotate('going to work', xy=(7, 0), xytext=(4, 230500), arrowprops={'arrowstyle': '->'}, fontsize=12)
    plt.annotate('getting off work', xy=(16, 0), xytext=(18, 230500), arrowprops={'arrowstyle': '->'}, fontsize=12)
    plt.savefig("../plots/accidents_per_hour_count_plot.png")

    """
    * From the above distribution we can observe that in the morning hours accidents occur more frequently during 
    6 - 9 Am period, where as in the evening hours they occur more frequently during 4 to 6 Pm.
    * We assume that those are hours of the day where people commute from school and work.
    * So higher measures must be taken during those times to reduce accidents.
    """

    print(data['Severity'].value_counts(), "\n")
    drawCountPlot(data, "Severity", "../plots/severity_count_plot.png", 'Accident Severity Frequency')

    """
    * We can observe that distribution of accident severity is skewed.
    * As there are a lot of accidents reported as level 2 severity.
    * And there are fewer number of accidents with level 3 and level 4 severity.
    * Level 1 severity has lowest number, maybe most of the mild accidents are not reported at all.
    """

    drawCatPlot(data, 'Start_Hour', 'Percent', True, 'Severity', "bar",
                'Severity of Accidents by Hour', "../plots/hour_severity_bar_plot.png")

    """
    * Accidents classified for various levels of severity remain fairly consistent for each hour of the day.
    * Level 2 accidents are the most common and level 1 are least common.
    * And there is no significant correlation observed between severity of accidents and the hour.
    """

    drawCatPlot(data, 'Severity', 'Percent', True, 'Sunrise_Sunset', "bar",
                'Severity of Accidents by Light', "../plots/sunrise_sunset_severity_bar_plot.png")

    """
    * Most of the severe accidents occur at night.
    * And most of the less severe accidents occur during the day.
    """

    drawBoxPlot(data, 'Severity', "../plots/severity_temp_box_plot.png", None, 'Temperature(F)')
    drawMeanPlot(data, 'Severity', 'Temperature(F)', "../plots/severity_temp_mean_plot.png")

    """
    * From the above plots we can observe that as the severity increases median and mean temperature decreases on overall distribution.
    * Mean and median temperatures of severity '2' and '3' accidents are almost same.
    * Where as significant difference is observed between median and mean temperatures of severity '1' and severity '4' accidents. 
    * Which suggests that accidents tend to be severe when temperatures are low or during winter.
    """

    drawBoxPlot(data, 'Severity', "../plots/severity_pressure_box_plot.png", None, 'Pressure(in)')
    drawMeanPlot(data, 'Severity', 'Pressure(in)', "../plots/severity_pressure_mean_plot.png")

    """No conclusion can be made as the ranges, median, mean pressure values of all accident severity is almost same."""

    drawBoxPlot(data, 'Severity', "../plots/severity_humidity_box_plot.png", None, 'Humidity(%)')
    drawMeanPlot(data, 'Severity', 'Humidity(%)', "../plots/severity_humidity_mean_plot.png")

    """
    * Median and mean humidity of accidents with severity '1' are lower when compared with other severity values.
    * Median and mean values for severity '2', '3', '4' are similar.
    """

    df = data.drop(
        columns=['Start_Time', 'End_Time', 'Start_Lat', 'Start_Lng', 'End_Lat', 'End_Lng', 'Distance(mi)',
                 'Description', 'State', 'Zipcode', 'Timezone', 'Street', 'Side', 'City', 'County', 'Airport_Code',
                 'Weather_Timestamp', 'Start_Year', 'Start_Month', 'Start_Day', 'Start_Weekday', 'Start_Hour',
                 'Duration(min)', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight'])

    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(22, 10))
    sns.heatmap(df[top_corr_features].corr().round(3), annot=True,
                cmap=sns.diverging_palette(93, 205, s=74, l=48, n=16)).set_title('Correlation matrix')
    plt.savefig("../plots/correlation_matrix.png")

    """
    * Above matrix displays the correlation values between some of the important features.
    * With this we are mainly interested in the correlation between severity and rest of the features.
    * And it is observed that none of the attributes have good correlation with the Severity.
    * Among the available attributes, "Wind_speed(mph) and "Junction" has the highest correlation Severity.
    * In between the attributes, "Traffic_Calming" and "Bump" records for the highest correlation.
    """

    drawMeanPlot(data, 'Severity', 'Wind_Speed(mph)', "../plots/severity_windSpeed_mean_plot.png")

    drawCatPlot(data, 'Severity', 'Percent', True, 'Junction', "bar",
                'Severity of Accidents at a junction', "../plots/junction_severity_bar_plot.png")

    drawCatPlot(data, 'Severity', 'Percent', True, 'Stop', "bar",
                'Severity of Accidents at a Stop', "../plots/stop_severity_bar_plot.png")

    drawCatPlot(data, 'Severity', 'Percent', True, 'Traffic_Signal', "bar",
                'Severity of Accidents at a Traffic_Signal', "../plots/trafficSignal_severity_bar_plot.png")

    drawCatPlot(data, 'Severity', 'Percent', True, 'Crossing', "bar",
                'Severity of Accidents at a Crossing', "../plots/crossing_severity_bar_plot.png")

    # sns.set()
    # plt.subplots(figsize=(20, 15))
    # plt.subplot(2, 2, 1)
    # sns.scatterplot(x='Visibility(mi)', y='Temperature(F)', hue='Severity', data=data).set_title(
    #     'Temperature vs Visibility')
    # plt.subplot(2, 2, 2)
    # sns.scatterplot(x='Visibility(mi)', y='Temperature(F)', hue='Severity',
    #                 data=data[(data['Visibility(mi)'] <= 5)]).set_title('Temperature vs Visibility')
    # plt.savefig("../plots/temp_visibility_scatter_plot.png")
    #
    # sns.set()
    # plt.subplots(figsize=(15, 7))
    # sns.scatterplot(x='Visibility(mi)', y='Wind_Speed(mph)', hue='Severity',
    #                 data=data[(data['Visibility(mi)'] < 18) & (data['Wind_Speed(mph)'] < 80)])\
    #     .set_title('Wind_Speed vs Visibility')
    # plt.savefig("../plots/windSpeed_visibility_scatter_plot.png")

    data.to_pickle("../data/processed_data.pkl")


if __name__ == "__main__":
    preProcess('D:\datasets/US_Accidents_Dec21_updated.csv')
