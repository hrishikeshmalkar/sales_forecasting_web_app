import streamlit as st
import pickle
import base64
#EDA pkg
import pandas as pd
import numpy as np

# Model Load/Save
from joblib import load
import joblib
import os

#Data Viz pkg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import ipywidgets as widgets
from IPython.display import display

import datetime
import statsmodels.tsa.api as smt

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor

import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam 
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import LSTM
from sklearn.model_selection import KFold, cross_val_score, train_test_split
## Functions
## Load css
def load_css(css_name):
	with open(css_name) as f:
		st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

## Load icon
def load_icon(name):
	st.markdown('<i class ="material-icons">{}</i>'.format(name), unsafe_allow_html=True)

## remote_css
def remote_css(url):
    st.markdown('<style src="{}"></style>'.format(url), unsafe_allow_html=True)

## icon-css
def icon_css(icone_name):
    remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

def monthly_sales(data):
    monthly_data = data.copy()
    monthly_data.date = monthly_data.date.apply(lambda x: str(x)[:-4])
    monthly_data = monthly_data.groupby('date')['sales'].sum().reset_index()
    monthly_data.date = pd.to_datetime(monthly_data.date)
    return monthly_data

def daily_sales(data):
    daily_data = data.copy()
    daily_data.date = daily_data.date.apply(lambda x: str(x)[:-4])
    daily_data=daily_data[['date','sales']]
    daily_data.date = pd.to_datetime(daily_data.date)
    return daily_data

# Duration of dataset
def sales_duration(data):
    data.date = pd.to_datetime(data.date)
    number_of_days = data.date.max() - data.date.min()
    number_of_months=number_of_days.days / 30
    number_of_years = number_of_days.days / 365
    return number_of_days.days, number_of_months, number_of_years

 #Determining Stationarity¶
def time_plot(data, x_col, y_col, title):
	fig, ax = plt.subplots(figsize=(15,5))
	sns.lineplot(x_col, y_col, data=data, ax=ax, color='mediumblue', label='Total Sales')
	second = data.groupby(data.date)[y_col].mean().reset_index()
	second.date = pd.to_datetime(second.date, format='%m')
	sns.lineplot(second.date, y_col, data=second, ax=ax, color='red', label='Mean Sales')
	ax.set(xlabel = "Date", ylabel = "Sales",title = title)
	sns.despine()

def get_diff(data):
    data['sales_diff'] = data.sales.diff()
    data = data.dropna()
    
    data.to_csv('./data/stationary_df.csv')
    return data

def plots(data, lags=None):
    
    # Convert dataframe to datetime index
    dt_data = data.set_index('date').drop('sales', axis=1)
    dt_data.dropna(axis=0)
    
    layout = (1, 3)
    raw  = plt.subplot2grid(layout, (0, 0))
    acf  = plt.subplot2grid(layout, (0, 1))
    pacf = plt.subplot2grid(layout, (0, 2))
    
    dt_data.plot(ax=raw, figsize=(12, 5), color='mediumblue')
    smt.graphics.plot_acf(dt_data, lags=lags, ax=acf, color='mediumblue')
    smt.graphics.plot_pacf(dt_data, lags=lags, ax=pacf, color='mediumblue')
    sns.despine()
    plt.tight_layout()

#creating dataframe for transformation from time series to supervised data
def generate_supervised(data):
    supervised_df = data.copy()
    
    #create column for each lag
    for i in range(1,13):
        col_name = 'lag_' + str(i)
        supervised_df[col_name] = supervised_df['sales_diff'].shift(i)
     
    supervised_df = supervised_df.dropna().reset_index(drop=True) #drop null values
    supervised_df.to_csv('./data/model_df.csv', index=False)
    return supervised_df

def generate_arima_data(data):
    dt_data = data.set_index('date').drop('sales', axis=1)
    dt_data.dropna(axis=0)
    
    dt_data.to_csv('./data/arima_df.csv')
    
    return dt_data

def train_test_split(data):
    data = data.drop(['sales','date'],axis=1)
    train, test = data[0:-8].values, data[-8:].values
    
    return train, test

def scale_data(train_set, test_set):
    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    
    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)
    
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()
    
    return X_train, y_train, X_test, y_test, scaler



##### Modelling Functions
def undo_scaling(y_pred, x_test, scaler_obj, lstm=False):  
    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
    
    if not lstm:
        x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    
    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index],x_test[index]],axis=1))
        
    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    
    #inverse transform
    pred_test_set_inverted = scaler_obj.inverse_transform(pred_test_set)
    
    return pred_test_set_inverted

def load_original_df():
    #load in original dataframe without scaling applied
    original_df = pd.read_excel(r'./data/BQ-Assignment-Data-Analytics.xlsx') 
    original_df.rename(columns=lambda x: x.replace(' ', '_'),inplace=True)
    original_df.columns= original_df.columns.str.lower()
    original_df.date = original_df.date.apply(lambda x: str(x)[:-4])
    original_df=original_df[['date','sales']]
    original_df.date = pd.to_datetime(original_df.date)
    return original_df

def predict_df(unscaled_predictions, original_df):
    #create dataframe that shows the predicted sales
    result_list = []
    sales_dates = list(original_df[-13:].date)
    act_sales = list(original_df[-13:].sales)
    
    for index in range(0,len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_sales[index])
        result_dict['date'] = sales_dates[index+1]
        result_list.append(result_dict)
        
    df_result = pd.DataFrame(result_list)
    
    return df_result


model_scores = {}

def get_scores(unscaled_df, original_df, model_name):
    rmse = np.sqrt(mean_squared_error(original_df.sales[-8:], unscaled_df.pred_value[-8:]))
    mae = mean_absolute_error(original_df.sales[-8:], unscaled_df.pred_value[-8:])
    r2 = r2_score(original_df.sales[-8:], unscaled_df.pred_value[-8:])
    model_scores[model_name] = [rmse, mae, r2]

    st.write(f"RMSE: {round(rmse,2)}")
    st.write(f"MAE: {round(mae,2)}")
    st.write(f"R2 Score: {round(r2,2)}")

def plot_results(results, original_df, model_name):

    fig, ax = plt.subplots(figsize=(15,5))
    sns.lineplot(original_df.date, original_df.sales, data=original_df, ax=ax, label='Original', color='mediumblue')
    sns.lineplot(results.date, results.pred_value, data=results, ax=ax, label='Predicted', color='Red')
    ax.set(xlabel = "Date",ylabel = "Sales",title = f"{model_name} Sales Forecasting Prediction")
    ax.legend()
    sns.despine()
    
def run_model(train_data, test_data, model, model_name):
    
    X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
    
    mod = model
    mod.fit(X_train, y_train)
    predictions = mod.predict(X_test)
    
    # Undo scaling to compare predictions against original data
    original_df = load_original_df()
    unscaled = undo_scaling(predictions, X_test, scaler_object)
    unscaled_df = predict_df(unscaled, original_df)
      
    get_scores(unscaled_df, original_df, model_name)
    
    plot_results(unscaled_df, original_df, model_name)

def lstm_model(train_data, test_data):
	X_train, y_train, X_test, y_test, scaler_object = scale_data(train_data, test_data)
	X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
	X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
	model = Sequential()
	model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
	model.add(Dense(1))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, shuffle=False)
	predictions = model.predict(X_test,batch_size=1)
	original_df = load_original_df()
	unscaled = undo_scaling(predictions, X_test, scaler_object, lstm=True)
	unscaled_df = predict_df(unscaled, original_df)
	get_scores(unscaled_df, original_df, 'LSTM')
	plot_results(unscaled_df, original_df, 'LSTM')


def create_results_df():
    results_dict = pickle.load(open("model_scores.p", "rb"))
    restults_df = pd.DataFrame.from_dict(results_dict, orient='index', columns=['RMSE', 'MAE','R2'])
    restults_df = restults_df.sort_values(by='RMSE', ascending=False).reset_index()
    return restults_df

def plot_resultss(results_df):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.lineplot(np.arange(len(results_df)), 'RMSE', data=results_df, ax=ax,label='RMSE', color='mediumblue')
    sns.lineplot(np.arange(len(results_df)), 'MAE', data=results_df, ax=ax, label='MAE', color='Cyan')
    plt.xticks(np.arange(len(results_df)))
    ax.set_xticklabels(results_df['index'])
    ax.set(xlabel = "Model",ylabel = "Scores",title = "Model Error Comparison")
    sns.despine()

#Data Exportion
def get_table_download_link(df):
	"""Generates a link allowing the data in a given panda dataframe to be downloaded in:  dataframe out: href string """
	csv = df.to_csv(index=False)
	b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
	return f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'


# Main
def main():
	""" Salary Prediction ML App with Streamlit """
	st.title("Sales Forecasting")
	st.text("ML Forecasting App with Streamlit")

	# Loading Dataset
	sales_data = pd.read_excel(r'BQ-Assignment-Data-Analytics.xlsx')
	sales_data.rename(columns=lambda x: x.replace(' ', '_'),inplace=True)
	sales_data.columns= sales_data.columns.str.lower()

	model_df = pd.read_csv('./data/model_df.csv')

	# Sidebar (TABS/ Menus)
	bars = ['EDA','Forecasting','About']
	choice = st.sidebar.selectbox("Choose Activity", bars)

	# Choice EDA
	if choice == 'EDA':
		st.subheader("Exploratory Data Analysis")
		load_css('icon.css') #function defines at the top
		load_icon('dashboard') #function defines at the top

		if st.checkbox("Show Dataset Preview"):
			num = st.number_input("Enter Number of Rows to Preview: ", value=5)
			st.dataframe(sales_data.head(num))
			st.markdown(get_table_download_link(sales_data), unsafe_allow_html=True)



		if st.checkbox("Shape of Dataset"):
			st.write(sales_data.shape)
			dim = st.radio("Show Dimensions by :",("Rows","Columns"))

			if dim == "Rows":
				st.text("Number of Rows :")
				st.write(sales_data.shape[0])
			elif dim == "Columns":
				st.text("Number of Columns :")
				st.write(sales_data.shape[1])

		if st.checkbox("Column Names"):
			all_columns = sales_data.columns.tolist()
			st.write(all_columns)

		if st.checkbox("Select Columns to Show"):
			all_columns = sales_data.columns.tolist()
			selected_col = st.multiselect("Select Columns", all_columns)
			new_col = sales_data[selected_col]
			st.dataframe(new_col)
		
		if st.checkbox("Select Rows to Show"):
			selected_index = st.multiselect("Select Rows: ", sales_data.head(15).index)
			selected_row = sales_data.loc[selected_index]
			st.dataframe(selected_row)

		if st.checkbox("Select Filteration"):
			
			item_type = st.selectbox("Show Data based on Item Type ", ('Select Item Type','All','Fruit','Vegetable'))
			
			if item_type == 'Select Item Type':
				st.text("Select particular Item Type")

			elif item_type == 'All':
				st.dataframe(sales_data)

			elif item_type == 'Fruit':
				st.dataframe(sales_data[sales_data.item_type == 'Fruit'])
				datadump=sales_data[sales_data.item_type == 'Fruit']
				st.markdown(get_table_download_link(datadump), unsafe_allow_html=True)


			elif item_type == 'Vegetable':
				st.dataframe(sales_data[sales_data.item_type == 'Vegetable'])
				datadump=sales_data[sales_data.item_type == 'Vegetable']
				st.markdown(get_table_download_link(datadump), unsafe_allow_html=True)



			item = st.selectbox("Show Data based on Items ", ('Select Items','All','Apple','Broccoli','Carrot','Cauliflower','Cherries','Lychee','Spinach','Strawberry'))
			
			if item == 'Select Items':
				st.text("Select particular Items")

			elif item == 'All':
				st.dataframe(sales_data)

			elif item == 'Apple':
				st.dataframe(sales_data[sales_data.item == 'Apple'])

			elif item == 'Broccoli':
				st.dataframe(sales_data[sales_data.item == 'Broccoli'])

			elif item == 'Carrot':
				st.dataframe(sales_data[sales_data.item == 'Carrot'])

			elif item == 'Cauliflower':
				st.dataframe(sales_data[sales_data.item == 'Cauliflower'])

			elif item == 'Cherries':
				st.dataframe(sales_data[sales_data.item == 'Cherries'])

			elif item == 'Lychee':
				st.dataframe(sales_data[sales_data.item == 'Lychee'])
				
			elif item == 'Spinach':
				st.dataframe(sales_data[sales_data.item == 'Spinach'])

			elif item == 'Strawberry':
				st.dataframe(sales_data[sales_data.item== 'Strawberry'])

		if st.checkbox("Show Info"):
			st.write(sales_data.dtypes)

		if st.checkbox("Show Description"):
			st.write(sales_data.describe())

		st.subheader("Data Visualization")
		load_css('icon.css')
		load_icon('show_charts')

		# Correlation plot with Matplotlib
		if st.checkbox("Correlation Plot [using Matplotlib]"):
			plt.matshow(sales_data	.corr())
			st.pyplot()

		# Correlation plot with Seaborn
		if st.checkbox("Correlation Plot with Annotation [using Seaborn]"):
			st.write(sns.heatmap(sales_data	.corr(), annot=True))
			st.pyplot()

		# Distplot with Seaborn
		if st.checkbox("Distplot of sales"):
			st.write(sns.distplot(sales_data['sales']))
			st.pyplot()

		# Donut chart
		if st.checkbox('Donut charts of Sales by Month'):
			month=['Jan','Feb','Mar','Apr','May']
			monthly_df = monthly_sales(sales_data)
			plt.pie(monthly_df.sales,labels=month,radius=1, frame=True)
			plt.pie([1],colors=['w'],radius=0.5)
			st.pyplot()

		# Duration of Dataset
		if st.checkbox('Duration of Sales Data'):
			days, month, year = sales_duration(sales_data)
			st.write("Days :",days)
			st.write("Months :",round(month,2))
			st.write("Years :",round(year,2))

		# Distribution of Sales Per Day
		if st.checkbox('Distribution of Sales per day'):
			fig, ax = plt.subplots(figsize=(7,4))
			plt.hist(sales_data.sales, color='mediumblue')
			ax.set(xlabel = "Sales Per day",ylabel = "Count",title = "Distribution of Sales Per Day")
			st.pyplot()

		# Distirubution of Sales per Item Sort ID
		if st.checkbox("Distribution of Sales per Item Sort ID"):
			by_item_sort_order = sales_data.groupby('item_sort_order')['sales'].sum().reset_index()
			fig, ax = plt.subplots(figsize=(7,4))
			sns.barplot(by_item_sort_order.item_sort_order, by_item_sort_order.sales, color='mediumblue')
			ax.set(xlabel = "item_sort_order ID",ylabel = "Number of Sales",title = "Total Sales Per Item ID")
			sns.despine()
			st.pyplot()

		# Distirubution of Sales per Items
		if st.checkbox("Distribution of Sales per Items"):
			by_item = sales_data.groupby('item')['sales'].sum().reset_index()
			fig, ax = plt.subplots(figsize=(7,4))
			sns.barplot(by_item.item, by_item.sales, color='mediumblue')
			ax.set(xlabel = "items", ylabel = "Number of Sales", title = "Total Sales Per Item Name")
			sns.despine()
			st.pyplot()

		# Monthly Avaerage Sales
		if st.checkbox("Monthly Average Sales"):
			avg_monthly_sales = monthly_df.sales.mean()
			a = f"Overall average monthly sales: ₹ {avg_monthly_sales}/-"
			st.markdown(a)

		# Monthly Sales before diff Transformation
		if st.checkbox('Monthly Sales before diff Transformation'):
			daily_df = daily_sales(sales_data)
			time_plot(daily_df, 'date', 'sales', 'Monthly Sales Before Diff Transformation')
			st.pyplot()

		# Monthly Sales after diff Transformation
		if st.checkbox('Monthly Sales after diff Transformation'):
			daily_df = daily_sales(sales_data)
			stationary_df = get_diff(daily_df)
			time_plot(stationary_df, 'date', 'sales_diff', 'Monthly Sales After Diff Transformation')
			st.pyplot()

		# Lags Observation
		if st.checkbox('Lags Observation'):
			daily_df = daily_sales(sales_data)
			stationary_df = get_diff(daily_df)
			plots(stationary_df, lags=24)
			st.pyplot()


	# Forecasting
	if choice == 'Forecasting':
		st.subheader("Sales Forecasting")
		load_css('icon.css') #function defines at the top
		load_icon('show_charts')

		st.markdown('## Dataset Modelling')
		st.markdown('### 1. Regressive Modelling ')

		if st.checkbox("Show lag Dataset"):
			daily_df = daily_sales(sales_data)
			stationary_df = get_diff(daily_df)
			model_df = generate_supervised(stationary_df)
			st.dataframe(model_df)
			st.markdown(get_table_download_link(model_df), unsafe_allow_html=True)

		st.markdown('### 2. Arima Modelling ')

		if st.checkbox("Show Arima-Model Dataset"):
			daily_df = daily_sales(sales_data)
			stationary_df = get_diff(daily_df)
			datetime_df = generate_arima_data(stationary_df)
			st.dataframe(datetime_df)
			st.markdown(get_table_download_link(datetime_df), unsafe_allow_html=True)

		
		st.markdown('## Modelling ')
		train, test = train_test_split(model_df)

		if st.checkbox("Shape of Train & Test Dataset"):
			
			st.write("Train Shape :",train.shape)
			st.write("Test Shape :",test.shape)


		X_train, y_train, X_test, y_test, scaler_object = scale_data(train, test)

		if st.checkbox("Forecasting: Using Linear Regression"):
			run_model(train, test, LinearRegression(), 'LinearRegression')
			st.pyplot()

		if st.checkbox("Forecasting: Using Random Forest Regressor"):
			run_model(train, test, RandomForestRegressor(n_estimators=100, max_depth=20), 'RandomForest')
			st.pyplot()

		if st.checkbox("Forecasting: Using XG Boost Regressor"):
			run_model(train, test, XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror'), 'XGBoost')
			st.pyplot()

		if st.checkbox("Forecasting: Using LSTM"):
			train, test = train_test_split(model_df)
			lstm_model(train, test)
			st.pyplot()

		# result
		st.markdown('## Modelling Results Comparison')
		if st.checkbox('Comapre Forecasting Results'):
			results = create_results_df()
			st.dataframe(results)
			st.markdown(get_table_download_link(results), unsafe_allow_html=True)

		if st.checkbox('Visualization of compared forecasting Results'):
			plot_resultss(results)
			st.pyplot()

			rf=25.375
			average_monthly_sales = 548.8 #see eda notebook
			percentage_off = round(rf/average_monthly_sales*100, 2)
			st.success(f"With Random Forest, prediction is within {percentage_off}% of the actual.")
			st.info("It seems that Random Forest is Performing so good rather than other models.")


			

			

	# ABOUT CHOICE
	if choice == 'About':
		st.subheader("About")
		st.markdown("""
			#### Sales Forecasting ML App
			##### Built with Streamlit

			#### By
			+ Hrishikesh Sharad Malkar
			
			""")


if __name__ == '__main__':
	main()
