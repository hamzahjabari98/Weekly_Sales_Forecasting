import streamlit as st
import pandas as pd
import numpy as np 
import plotly.express as px 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
import joblib
from datetime import datetime, timedelta
import calendar
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor


class DataLoadMergePreprocess:
    def __init__(self, features_path='features.csv', stores_path='stores.csv', train_path='train.csv', test_path='test.csv',
                 selected_features=['Store', 'Dept', 'Type', 'Size', 'Week', 'Thanksgiving_Day']):
        self.features_path = features_path
        self.stores_path = stores_path
        self.train_path = train_path
        self.test_path = test_path
        self.selected_features = selected_features
        self.df = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.scaler = None

    def load_merge_data(self):
        """
        Load and merge data from features, stores, and train CSV files into one dataframe.
        """
        # Load data from CSV files
        self.features = pd.read_csv(self.features_path)
        self.stores = pd.read_csv(self.stores_path)
        self.train = pd.read_csv(self.train_path)
        self.test = pd.read_csv(self.test_path)

        # Fill missing values in the 'MarkDown' columns with zeros
        MarkDown_column = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
        for column in MarkDown_column:
            median_value = self.features[column].median()
            self.features[column] = self.features[column].fillna(median_value)

        # Fill missing values in 'CPI' and 'Unemployment' columns with the last recorded non-null value
        other = ['CPI', 'Unemployment']
        for column in other:
            self.features[column] = self.features[column].fillna(method='ffill')

        # Merge the train, stores, and features dataframes based on common columns
        train_stores = pd.merge(self.train, self.stores, on='Store')
        test_stores = pd.merge(self.test, self.stores, on='Store')

        self.df = pd.merge(train_stores, self.features, on=['Store', 'Date', 'IsHoliday'])
        self.test_data = pd.merge(test_stores, self.features, on=['Store', 'Date', 'IsHoliday'])

    def preprocess(self, holiday_columns=True, encoding=True, outlier=True, scale_test=True, scale_split=True):
        if self.df is None:
            raise Exception("Data has not been loaded and merged yet.")

        if self.test_data is None:
            raise Exception("Data has not been loaded and merged yet.")

        # Stripping whitespace from column names
        self.df.columns = self.df.columns.str.strip()
        self.test_data.columns = self.test_data.columns.str.strip()

        # Transforming the date column into datetime and extracting all the needed features from the date column
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Week'] = self.df['Date'].dt.isocalendar().week.astype(int)
        self.df['Year'] = self.df['Date'].dt.year

        self.test_data['Date'] = pd.to_datetime(self.test_data['Date'])
        self.test_data['Week'] = self.test_data['Date'].dt.isocalendar().week.astype(int)

        # Taking the absolute value of the target column
        self.df['Weekly_Sales'] = self.df['Weekly_Sales'].abs().values

        if holiday_columns:
            # Extracting holiday names from IsHoliday column
            self.df = self.add_holiday_columns(self.df)
            self.test_data = self.add_holiday_columns(self.test_data)

        self.df.sort_values(by='Date', ascending=True, inplace=True)
        self.df.set_index('Date', inplace=True)

        self.test_data.sort_values(by='Date', ascending=True, inplace=True)
        self.test_data.set_index('Date', inplace=True)

        if outlier:
            self.df = self.outlier_treatment(self.df)
            self.test_data = self.outlier_treatment(self.test_data)

        if encoding:
            self.df = self.encoding(self.df)
            self.test_data = self.encoding(self.test_data)

        if scale_split:
            self.X, self.y, self.X_train, self.X_test, self.y_train, self.y_test, self.scaler = self.split_scale(self.df)

        if scale_test:
            num_col = ['Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
                       'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']

            self.test_data[num_col] = self.scaler.transform(self.test_data[num_col])

        if self.selected_features:
            self.test_data = self.test_data[self.selected_features]
            self.X = self.X[self.selected_features]
            self.X_train = self.X_train[self.selected_features]
            self.X_test = self.X_test[self.selected_features]

    def encoding(self, df, bool_col=['IsHoliday', 'Super_Bowl', 'Labour_Day', 'Thanksgiving_Day', 'Christmas'],
                 cat_col='Type', bool_enc_type='numeric', cat_enc_type='numeric'):
        if self.df is None:
            raise Exception("Data has not been loaded and merged yet.")

        if bool_enc_type == 'numeric':
            df[bool_col] = np.where(df[bool_col] == True, 1, 0)
        elif bool_enc_type == 'onehotencode':
            df = pd.get_dummies(df, columns=[bool_col])
        elif bool_enc_type == 'labelencode':
            le = LabelEncoder()
            df[bool_col] = le.fit_transform(df[bool_col])
        else:
            raise ValueError("Invalid boolean encoding type. Please choose 'numeric', 'onehotencode', or 'labelencode'.")

        if cat_enc_type == 'numeric':
            df[cat_col] = np.where(df[cat_col] == 'A', 3, np.where(df[cat_col] == 'B', 2, 1))
        elif cat_enc_type == 'onehotencode':
            df = pd.get_dummies(df, columns=[cat_col])
        elif cat_enc_type == 'ordinalencode':
            oe = OrdinalEncoder()
            df[cat_col] = oe.fit_transform(df[cat_col])
        else:
            raise ValueError("Invalid categorical encoding type. Please choose 'numeric', 'onehotencode' or 'ordinalencode'.")
        return df

    def outlier_treatment(self, df):
        whisker_multiplier = 1.5
        col_num = ['Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
                   'CPI', 'Unemployment']
        for column in col_num:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_whisker = Q1 - whisker_multiplier * IQR
            upper_whisker = Q3 + whisker_multiplier * IQR

            df[column] = np.where(df[column] < lower_whisker, lower_whisker, df[column])
            df[column] = np.where(df[column] > upper_whisker, upper_whisker, df[column])

        return df

    def add_holiday_columns(self, df):
        if self.df is None:
            raise Exception("Data has not been loaded and merged yet.")

        # Extracting a column for each holiday that exists in the IsHoliday column
        Super_Bowl = pd.to_datetime(["2010-02-12", "2011-02-11", "2012-02-10", "2013-02-08"])
        Labour_Day = pd.to_datetime(["2010-09-10", "2011-09-09", "2012-09-07"])
        Thanksgiving_Day = pd.to_datetime(["2010-11-26", "2011-11-25", '2012-11-23'])
        Christmas = pd.to_datetime(["2010-12-31", "2011-12-30", '2012-12-28'])
        df["Super_Bowl"] = df["Date"].isin(Super_Bowl)
        df["Labour_Day"] = df["Date"].isin(Labour_Day)
        df["Thanksgiving_Day"] = df["Date"].isin(Thanksgiving_Day)
        df["Christmas"] = df["Date"].isin(Christmas)
        return df

    def split_scale(self, df, scale=True, scaling_method='standard', year_train_start=2010, year_train_end=2012,
                    year_test=2012):
        X_train = df[(df['Year'] >= year_train_start) & (df['Year'] < year_train_end)].drop('Weekly_Sales', axis=1)
        y_train = df[(df['Year'] >= year_train_start) & (df['Year'] < year_train_end)]['Weekly_Sales']

        X_test = df[df['Year'] == year_test].drop('Weekly_Sales', axis=1)
        y_test = df[df['Year'] == year_test]['Weekly_Sales']

        if scale:
            num_col = ['Size', 'Temperature', 'Fuel_Price', 'MarkDown1', 'MarkDown2',
                       'MarkDown3', 'MarkDown4', 'MarkDown5', 'CPI', 'Unemployment']

            if scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError("Invalid scaling method. Please choose 'minmax' or 'standard'.")

            X_train[num_col] = scaler.fit_transform(X_train[num_col])
            X_test[num_col] = scaler.transform(X_test[num_col])

        return df.drop('Weekly_Sales', axis=1), df['Weekly_Sales'], X_train, X_test, y_train, y_test, scaler


@st.cache_data
def data_loader(load_test=False, load_train_test=False, load_df=False):
    data_processor = DataLoadMergePreprocess()
    data_processor.load_merge_data()
    data_processor.preprocess()

    if load_test and load_train_test:
        return data_processor.test_data, data_processor.X_train, data_processor.X_test, data_processor.y_train, data_processor.y_test

    elif load_test:
        return data_processor.test_data

    elif load_df:
        return data_processor.df


def load_data(data_loader, data_type):
    test_data, X_train, X_test, y_train, y_test = data_loader(load_test=True, load_train_test=True)

    if data_type == 'Testing Data without Target':
        return test_data

    elif data_type == 'Training and testing data':
        return X_train, X_test, y_train, y_test

    elif data_type == 'All Data':
        return test_data, X_train, X_test, y_train, y_test

@st.cache_data
def data_resample(y):
    return y.resample('W').mean()

def time_series_plot(y):
    fig = go.Figure(data=go.Scatter(x=data_resample(y).index, y=data_resample(y).values))
    fig.update_layout(
        title='Average Sales - Weekly',
        xaxis_title='Date',
        yaxis_title='Average Sales'
    )
    fig.update_traces(marker=dict(line=dict(color='navy', width=2)))
    st.plotly_chart(fig, clear_figure=True, use_column_width=True,)

def resample_plot(arr1, arr2, only_resample=False, only_plot=True, resample_and_plot=False):
    if only_resample:
        return data_resample(arr1), data_resample(arr2)

    elif only_plot:
        with st.expander("Plot Prediction"):
            col1, col2, col3 = st.columns([4, 10, 3.5])
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data_resample(arr1).index, y=data_resample(arr1).values, name='Train', marker=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data_resample(arr2).index, y=data_resample(arr2).values, name='Test', marker=dict(color='red')))
                fig.update_layout(
                    title='Weekly Sales',
                    xaxis_title='Date',
                    yaxis_title='Average Sales',
                    font=dict(size=14),
                    showlegend=True
                )
                st.plotly_chart(fig, clear_figure=True, use_column_width=True,)

    elif resample_and_plot:
        with st.expander("Plot Prediction"):
            col1, col2, col3 = st.columns([4, 10, 3.5])
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data_resample(arr1).index, y=data_resample(arr1).values, name='Train', marker=dict(color='blue')))
                fig.add_trace(go.Scatter(x=data_resample(arr2).index, y=data_resample(arr2).values, name='Test', marker=dict(color='red')))
                fig.update_layout(
                    title='Weekly Sales',
                    xaxis_title='Date',
                    yaxis_title='Average Sales',
                    font=dict(size=14),
                    showlegend=True
                )
                st.plotly_chart(fig, clear_figure=True, use_column_width=True,)
        return data_resample(arr1), data_resample(arr2)

def model_load(selected_model):
    with open(f'{selected_model}_model.joblib', 'rb') as file:
        return joblib.load(file)

def model_predict(model, X):
    return model.predict(X)

def plot_predictions(actual_values, predicted_values):
    with st.expander("Plot Prediction"):
        col1, col2, col3 = st.columns([4, 10, 3.5])
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=predicted_values.index, y=predicted_values.iloc[:, 0], name='Predicted'))
            fig.add_trace(go.Scatter(x=actual_values.index, y=actual_values.iloc[:, 0], name='Actual'))
            fig.update_layout(title='Actual vs Predicted Sales', xaxis_title='Date', yaxis_title='Sales', legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            st.plotly_chart(fig, clear_figure=True, use_column_width=True,)

def plot_prediction_no_target(predicted_values):
    with st.expander("Plot Prediction"):
        col1, col2, col3 = st.columns([4, 10, 3.5])
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=predicted_values.index, y=predicted_values.iloc[:, 0], name='Predicted'))
            fig.update_layout(title='Predicted Sales', xaxis_title='Date', yaxis_title='Sales', legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            st.plotly_chart(fig, use_column_width=True,)

def plot_target_no_target(y, pred_target, actual_data, pred_no_target):
    colors = ['blue', 'green', 'red', 'orange']  # Add more colors if needed

    with st.expander("Plot Prediction"):
        col1, col2, col3 = st.columns([3.5, 10, 1.5])
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y.index, y=y.iloc[:, 0], name='Training Data', line=dict(color=colors[0])))
            fig.add_trace(go.Scatter(x=pred_target.index, y=pred_target.iloc[:, 0], name='Prediction of data with target', line=dict(color=colors[1])))
            fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data.iloc[:, 0], name='Actual Data', line=dict(color=colors[2])))
            fig.add_trace(go.Scatter(x=pred_no_target.index, y=pred_no_target.iloc[:, 0], name='Prediction of data without target', line=dict(color=colors[3])))

            fig.update_layout(title='Predicted Sales', xaxis_title='Date', yaxis_title='Sales', legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ))
            st.plotly_chart(fig, use_column_width=True, width=400)
    

def main():
    # Setting the theme and layout
    st.set_page_config(
        page_title="Weekly Sales Forecasting",
        page_icon=":smiley:",
        layout="wide",
        initial_sidebar_state="auto"
    )
    st.markdown(
    """
    <style>
        button[title^=Exit]+div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
    """, unsafe_allow_html=True
)
    header_container = st.container()

    col1, col2 = header_container.columns([3, 7])

    # Display the logo
    col1.image('Walmart_logo_transparent.png', width=200)

    # Display the title
    col2.title('Walmart Sales Forecasting')

    col1, col2 = st.columns(2)
    model_names = ['Models', 'RandomForest', 'XGBRegressor', 'CatBoostRegressor', 'Stacked']
    selected_model = col1.selectbox('Select Model', model_names, key='model_selectbox')

    if selected_model != 'Models':
        date_prediction_type = ['Prediction Type', 'Weekly', 'Monthly', 'Yearly']
        selected_date_option = col2.selectbox('Select Prediction Type', date_prediction_type, key='prediction_selectbox')

        if selected_date_option != 'Prediction Type':

            if selected_date_option == 'Weekly':
                test_data, X_train, X_test, y_train, y_test = load_data(data_loader, 'All Data')
                dates = test_data.index.unique().to_list() + y_test.index.unique().to_list()
                formatted_dates = date_series = pd.Series(sorted(dates))
                min_date = date_series.min()
                max_date = date_series.max()
                
                week_options = []
                current_date = min_date
                while current_date <= max_date:
                    week_options.append(f"Week {current_date.strftime('%U')} - {current_date.strftime('%Y')}")
                    current_date += timedelta(weeks=1)

                # Generate week numbers from the formatted dates
                week_numbers = formatted_dates.dt.strftime("Week %U - %Y")

                # Map week numbers to a numerical range
                week_mapping = {week: i for i, week in enumerate(week_numbers)}

                # Slider for choosing week numbers
                selected_index = st.slider('Select a Week', 0, len(week_mapping) - 1)

                # Retrieve the selected week number based on the chosen index
                selected_week = week_numbers[selected_index]

                # Retrieve the selected date based on the chosen week number
                selected_date = formatted_dates[week_numbers == selected_week].iloc[0]

                # Format the selected date as day-month-year
                selected_date_str = selected_date.strftime("%d-%m-%Y")
                # Filter the data based on the selected date from test_data
                selected_test_data = test_data[test_data.index.date == selected_date.date()]

                # Filter the data based on the selected date from X_test and y_test
                selected_X_test = X_test[X_test.index.date == selected_date.date()]
                selected_y_test = y_test[y_test.index.date == selected_date.date()]

                if selected_model == 'RandomForest':
                    model = model_load(selected_model)

                    if selected_X_test.empty:
                        y_pred = model_predict(model, selected_test_data)
                        y_pred = pd.DataFrame(y_pred,index = selected_test_data.index,columns=['Prediction'])
                        st.markdown(f"## Results of Week ({selected_date_str})")
                        st.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                    else:
                        y_pred = model_predict(model, selected_X_test)
                        y_pred = pd.DataFrame(y_pred,index = selected_X_test.index,columns=['Prediction'])

                        y_actual = pd.DataFrame(selected_y_test, index=selected_X_test.index, columns=['Actual'])
                        result = pd.concat([data_resample(y_pred), data_resample(y_actual)], axis=1)

                        # Display results
                        st.markdown(f"## Results of Week ({selected_date_str})")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                        col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                        col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                        col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')


                elif selected_model == 'XGBRegressor':
                    model = model_load(selected_model)

                    if selected_X_test.empty:
                        y_pred = model_predict(model, selected_test_data)
                        y_pred = pd.DataFrame(y_pred,index = selected_test_data.index,columns=['Prediction'])

                        st.markdown(f"## Results of Week ({selected_date_str})")
                        st.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                    else:
                        y_pred = model_predict(model, selected_X_test)
                        y_pred = pd.DataFrame(y_pred,index = selected_X_test.index,columns=['Prediction'])

                        y_actual = pd.DataFrame(selected_y_test, index=selected_X_test.index, columns=['Actual'])
                        result = pd.concat([data_resample(y_pred), data_resample(y_actual)], axis=1)

                        # Display results
                        st.markdown(f"## Results of Week ({selected_date_str})")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                        col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                        col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                        col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                elif selected_model == 'CatBoostRegressor':
                    model = model_load(selected_model)

                    if selected_X_test.empty:
                        y_pred = model_predict(model, selected_test_data)
                        y_pred = pd.DataFrame(y_pred,index = selected_test_data.index,columns=['Prediction'])
                        st.markdown(f"## Results of Week ({selected_date_str})")
                        st.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                    else:
                        y_pred = model_predict(model, selected_X_test)
                        y_pred = pd.DataFrame(y_pred,index = selected_X_test.index,columns=['Prediction'])

                        y_actual = pd.DataFrame(selected_y_test, index=selected_X_test.index, columns=['Actual'])
                        result = pd.concat([data_resample(y_pred), data_resample(y_actual)], axis=1)

                        # Display results
                        st.markdown(f"## Results of Week ({selected_date_str})")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                        col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                        col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                        col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                        
                elif selected_model == 'Stacked':
                    model = model_load(selected_model)

                    if selected_X_test.empty:
                        y_pred = model_predict(model, selected_test_data)
                        y_pred = pd.DataFrame(y_pred,index = selected_test_data.index,columns=['Prediction'])
                        st.markdown(f"## Results of Week ({selected_date_str})")
                        st.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                    else:
                        y_pred = model_predict(model, selected_X_test)
                        y_pred = pd.DataFrame(y_pred,index = selected_X_test.index,columns=['Prediction'])
                        y_actual = pd.DataFrame(selected_y_test, index=selected_X_test.index, columns=['Actual'])

                        # Display results
                        st.markdown(f"## Results of Week ({selected_date_str})")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                        col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                        col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                        col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

            if selected_date_option == 'Monthly':
                test_data, X_train, X_test, y_train, y_test = load_data(data_loader, 'All Data')
                dates = test_data.index.unique().to_list() + y_test.index.unique().to_list()
                formatted_dates = pd.Series(sorted(dates))
                min_date = formatted_dates.min()
                max_date = formatted_dates.max()

                # Generate month options from the formatted dates
                month_options = pd.date_range(min_date, max_date, freq='MS').strftime('%B %Y').tolist()

                # Checkboxes for selecting months
                selected_months = st.multiselect('Select Months', month_options)
                if not selected_months:
                    st.warning("Please select a month or an interval of months.")
                else:
                    # Filter the data based on the selected months
                    selected_test_data = test_data[test_data.index.strftime('%B %Y').isin(selected_months)]
                    selected_X_test = X_test[X_test.index.strftime('%B %Y').isin(selected_months)]
                    selected_y_test = y_test[y_test.index.strftime('%B %Y').isin(selected_months)]

                    if selected_model == 'RandomForest':
                        model = model_load(selected_model)

                        if selected_X_test.empty:
                            y_pred = model_predict(model, selected_test_data)
                            y_pred = pd.DataFrame(y_pred,index = selected_test_data.index,columns=['Prediction'])

                            if len(selected_months) == 1:
                                interval_string = f"{selected_months[0]}"
                            else:
                                interval_string = f"{selected_months[0]} - {selected_months[-1]}"
                            
                            st.subheader(f"Results for {interval_string}")
                            st.metric("Mean Monthly Sales Prediction", round(float(data_resample(y_pred).mean()), 2))
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            plot_prediction_no_target(y_pred_df)

                        else:
                            y_pred = model_predict(model, selected_X_test)
                            y_pred = pd.DataFrame(y_pred, index = selected_X_test.index,columns=['Prediction'])
                            
                            # Display results
                            if len(selected_months) == 1:
                                interval_string = f"{selected_months[0]}"
                            else:
                                interval_string = f"{selected_months[0]} - {selected_months[-1]}"

                            st.subheader(f"Results for {interval_string}")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                            col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                            col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                            col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                            y_pred_df = pd.DataFrame(data_resample(y_pred))
                            selected_y_test_df = pd.DataFrame(data_resample(selected_y_test))

                            plot_predictions(selected_y_test_df, y_pred_df)

                    elif selected_model == 'XGBRegressor':
                        model = model_load(selected_model)

                        if selected_X_test.empty:
                            y_pred = model_predict(model, selected_test_data)
                            y_pred = pd.DataFrame(y_pred,index = selected_test_data.index,columns=['Prediction'])
                            if len(selected_months) == 1:
                                interval_string = f"{selected_months[0]}"
                            else:
                                interval_string = f"{selected_months[0]} - {selected_months[-1]}"
                            st.subheader(f"Results for {interval_string}")
                            st.metric("Mean Monthly Sales Prediction", round(float(data_resample(y_pred).mean()), 2))
                            y_pred_df = pd.DataFrame(data_resample(y_pred))
                            plot_prediction_no_target(y_pred_df)

                        else:
                            y_pred = model_predict(model, selected_X_test)
                            y_pred = pd.DataFrame(y_pred, index = selected_X_test.index,columns=['Prediction'])
                            # Display results
                            if len(selected_months) == 1:
                                interval_string = f"{selected_months[0]}"
                            else:
                                interval_string = f"{selected_months[0]} - {selected_months[-1]}"
                            st.subheader(f"Results for {interval_string}")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                            col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                            col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                            col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                            y_pred_df = pd.DataFrame(data_resample(y_pred))
                            selected_y_test_df = pd.DataFrame(data_resample(selected_y_test))

                            plot_predictions(selected_y_test_df, y_pred_df)

                    elif selected_model == 'CatBoostRegressor':
                        model = model_load(selected_model)

                        if selected_X_test.empty:
                            y_pred = model_predict(model, selected_test_data)
                            y_pred = pd.DataFrame(y_pred,index = selected_test_data.index,columns=['Prediction'])
                            if len(selected_months) == 1:
                                interval_string = f"{selected_months[0]}"
                            else:
                                interval_string = f"{selected_months[0]} - {selected_months[-1]}"
                            st.subheader(f"Results for {interval_string}")
                            st.metric("Mean Monthly Sales Prediction", round(float(data_resample(y_pred).mean()), 2))
                            y_pred_df = pd.DataFrame(data_resample(y_pred))
                            
                            plot_prediction_no_target(y_pred_df)

                        else:
                            y_pred = model_predict(model, selected_X_test)
                            y_pred = pd.DataFrame(y_pred, index = selected_X_test.index,columns=['Prediction'])
                            # Display results
                            if len(selected_months) == 1:
                                interval_string = f"{selected_months[0]}"
                            else:
                                interval_string = f"{selected_months[0]} - {selected_months[-1]}"
                            st.subheader(f"Results for {interval_string}")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                            col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                            col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                            col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                            y_pred_df = pd.DataFrame(data_resample(y_pred))
                            selected_y_test_df = pd.DataFrame(data_resample(selected_y_test))

                            plot_predictions(selected_y_test_df, y_pred_df)

                            
                    elif selected_model == 'Stacked':
                        model = model_load(selected_model)

                        if selected_X_test.empty:
                            y_pred = model_predict(model, selected_test_data)
                            y_pred = pd.DataFrame(y_pred,index = selected_test_data.index,columns=['Prediction'])
                            if len(selected_months) == 1:
                                interval_string = f"{selected_months[0]}"
                            else:
                                interval_string = f"{selected_months[0]} - {selected_months[-1]}"
                            st.subheader(f"Results for {interval_string}")
                            st.metric("Mean Monthly Sales Prediction", round(float(data_resample(y_pred).mean()), 2))
                            y_pred_df = pd.DataFrame(data_resample(y_pred))
                        
                            plot_prediction_no_target(y_pred_df)

                        else:
                            y_pred = model_predict(model, selected_X_test)
                            y_pred = pd.DataFrame(y_pred, index = selected_X_test.index,columns=['Prediction'])
                            # Display results
                            if len(selected_months) == 1:
                                interval_string = f"{selected_months[0]}"
                            else:
                                interval_string = f"{selected_months[0]} - {selected_months[-1]}"
                            st.subheader(f"Results for {interval_string}")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                            col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                            col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                            col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')
                    
                            y_pred_df = pd.DataFrame(data_resample(y_pred))
                            selected_y_test_df = pd.DataFrame(data_resample(selected_y_test))

                            plot_predictions(selected_y_test_df, y_pred_df)

            if selected_date_option == 'Yearly':
                test_data, X_train, X_test, y_train, y_test = load_data(data_loader, 'All Data')
                dates = test_data.index.unique().to_list() + y_test.index.unique().to_list()
                formatted_dates = pd.Series(sorted(dates))
                unique_years = formatted_dates.dt.year.unique()

                selected_years = st.multiselect('Select Years', unique_years.tolist())
                if not selected_years:
                    st.warning("Please select a year or an interval of years.")
                else:
                    # Filter the data based on the selected years
                    selected_test_data = test_data[test_data.index.year.isin(selected_years)]
                    selected_X_test = X_test[X_test.index.year.isin(selected_years)]
                    selected_y_test = y_test[y_test.index.year.isin(selected_years)]

                    if selected_model == 'RandomForest':
                        model = model_load(selected_model)

                        if selected_X_test.empty:
                            y_pred = model_predict(model, selected_test_data)
                            y_pred = pd.DataFrame(y_pred, index=selected_test_data.index, columns=['Prediction'])

                            if selected_years[0] == selected_years[-1]:
                                interval_string = f"{selected_years[0]}"
                            else:
                                interval_string = f"{selected_years[0]} - {selected_years[-1]}"

                            st.subheader(f"Results for {interval_string}")
                            st.metric("Mean Sales Prediction", round(float(data_resample(y_pred).mean()), 2))
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            plot_prediction_no_target(y_pred_df)   

                        else:
                            y_pred = model_predict(model, selected_X_test)                            
                            y_pred = pd.DataFrame(y_pred, index = selected_X_test.index,columns=['Prediction'])
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            y_pred_no_label = model_predict(model, selected_test_data) 
                            y_pred_no_label = pd.DataFrame(y_pred_no_label, index=selected_test_data.index, columns=['Prediction']) 
                            y_pred_no_label_df = pd.DataFrame(data_resample(y_pred_no_label))

                            actual_data = pd.DataFrame(selected_y_test.values, index=selected_X_test.index, columns=['Weekly_Sales'])
                            actual_data_df = pd.DataFrame(data_resample(actual_data))

                            y = pd.DataFrame(y_train.values, index=y_train.index, columns=['Weekly_Sales'])
                            y_data = pd.DataFrame(data_resample(y))

                            if selected_years[0] == selected_years[-1]:
                                interval_string = f"{selected_years[0]}"
                            else:
                                interval_string = f"{selected_years[0]} - {selected_years[-1]}"
                            
                            st.subheader(f"Prediction Results for Data with Target in the {interval_string} Period")
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                            col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                            col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                            col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                            st.subheader(f"Prediction Results for Data with No Target in the {interval_string} Period")
                            st.metric("Mean Sales Prediction", round(float(data_resample(y_pred_no_label_df).mean()), 2))

                            plot_target_no_target(y_data, y_pred_df, actual_data_df, y_pred_no_label_df)


                    elif selected_model == 'XGBRegressor':
                        model = model_load(selected_model)

                        if selected_X_test.empty:
                            y_pred = model_predict(model, selected_test_data)
                            y_pred = pd.DataFrame(y_pred, index=selected_test_data.index, columns=['Prediction'])

                            if selected_years[0] == selected_years[-1]:
                                interval_string = f"{selected_years[0]}"
                            else:
                                interval_string = f"{selected_years[0]} - {selected_years[-1]}"

                            st.subheader(f"Results for {interval_string}")
                            st.metric("Mean Sales Prediction", round(float(data_resample(y_pred).mean()), 2))
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            plot_prediction_no_target(y_pred_df)   

                        else:
                            y_pred = model_predict(model, selected_X_test)                            
                            y_pred = pd.DataFrame(y_pred, index = selected_X_test.index,columns=['Prediction'])
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            y_pred_no_label = model_predict(model, selected_test_data) 
                            y_pred_no_label = pd.DataFrame(y_pred_no_label, index=selected_test_data.index, columns=['Prediction']) 
                            y_pred_no_label_df = pd.DataFrame(data_resample(y_pred_no_label))

                            actual_data = pd.DataFrame(selected_y_test.values, index=selected_X_test.index, columns=['Weekly_Sales'])
                            actual_data_df = pd.DataFrame(data_resample(actual_data))
                            
                            y = pd.DataFrame(y_train.values, index=y_train.index, columns=['Weekly_Sales'])
                            y_data = pd.DataFrame(data_resample(y))

                            if selected_years[0] == selected_years[-1]:
                                interval_string = f"{selected_years[0]}"
                            else:
                                interval_string = f"{selected_years[0]} - {selected_years[-1]}"
                            
                            st.subheader(f"Prediction Results for Data with Target in the {interval_string} Period") 
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                            col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                            col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                            col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                            st.subheader(f"Prediction Results for Data with No Target in the {interval_string} Period")
                            st.metric("Mean Sales Prediction", round(float(data_resample(y_pred_no_label_df).mean()), 2))

                            plot_target_no_target(y_data, y_pred_df, actual_data_df, y_pred_no_label_df)


                    elif selected_model == 'CatBoostRegressor':
                        model = model_load(selected_model)

                        if selected_X_test.empty:
                            y_pred = model_predict(model, selected_test_data)
                            y_pred = pd.DataFrame(y_pred, index=selected_test_data.index, columns=['Prediction'])

                            if selected_years[0] == selected_years[-1]:
                                interval_string = f"{selected_years[0]}"
                            else:
                                interval_string = f"{selected_years[0]} - {selected_years[-1]}"

                            st.subheader(f"Results for {interval_string}")
                            st.metric("Mean Sales Prediction", round(float(data_resample(y_pred).mean()), 2))
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            plot_prediction_no_target(y_pred_df)   

                        else:
                            y_pred = model_predict(model, selected_X_test)                            
                            y_pred = pd.DataFrame(y_pred, index = selected_X_test.index,columns=['Prediction'])
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            y_pred_no_label = model_predict(model, selected_test_data) 
                            y_pred_no_label = pd.DataFrame(y_pred_no_label, index=selected_test_data.index, columns=['Prediction']) 
                            y_pred_no_label_df = pd.DataFrame(data_resample(y_pred_no_label))

                            actual_data = pd.DataFrame(selected_y_test.values, index=selected_X_test.index, columns=['Weekly_Sales'])
                            actual_data_df = pd.DataFrame(data_resample(actual_data))

                            y = pd.DataFrame(y_train.values, index=y_train.index, columns=['Weekly_Sales'])
                            y_data = pd.DataFrame(data_resample(y))

                            if selected_years[0] == selected_years[-1]:
                                interval_string = f"{selected_years[0]}"
                            else:
                                interval_string = f"{selected_years[0]} - {selected_years[-1]}"
                            
                            st.subheader(f"Prediction Results for Data with Target in the {interval_string} Period")
 
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                            col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                            col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                            col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                            st.subheader(f"Prediction Results for Data with No Target in the {interval_string} Period")
                            st.metric("Mean Sales Prediction", round(float(data_resample(y_pred_no_label_df).mean()), 2))

                            plot_target_no_target(y_data, y_pred_df, actual_data_df, y_pred_no_label_df)

                    elif selected_model == 'Stacked':
                        model = model_load(selected_model)

                        if selected_X_test.empty:
                            y_pred = model_predict(model, selected_test_data)
                            y_pred = pd.DataFrame(y_pred, index=selected_test_data.index, columns=['Prediction'])

                            if selected_years[0] == selected_years[-1]:
                                interval_string = f"{selected_years[0]}"
                            else:
                                interval_string = f"{selected_years[0]} - {selected_years[-1]}"

                            st.subheader(f"Results for {interval_string}")
                            st.metric("Mean Sales Prediction", round(float(data_resample(y_pred).mean()), 2))
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            plot_prediction_no_target(y_pred_df)   

                        else:
                            y_pred = model_predict(model, selected_X_test)                            
                            y_pred = pd.DataFrame(y_pred, index = selected_X_test.index,columns=['Prediction'])
                            y_pred_df = pd.DataFrame(data_resample(y_pred))

                            y_pred_no_label = model_predict(model, selected_test_data) 
                            y_pred_no_label = pd.DataFrame(y_pred_no_label, index=selected_test_data.index, columns=['Prediction']) 
                            y_pred_no_label_df = pd.DataFrame(data_resample(y_pred_no_label))

                            actual_data = pd.DataFrame(selected_y_test.values, index=selected_X_test.index, columns=['Weekly_Sales'])
                            actual_data_df = pd.DataFrame(data_resample(actual_data))

                            y = pd.DataFrame(y_train.values, index=y_train.index, columns=['Weekly_Sales'])
                            y_data = pd.DataFrame(data_resample(y))

                            if selected_years[0] == selected_years[-1]:
                                interval_string = f"{selected_years[0]}"
                            else:
                                interval_string = f"{selected_years[0]} - {selected_years[-1]}"
                            
                            st.subheader(f"Prediction Results for Data with Target in the {interval_string} Period")

                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Mean Weekly Sales Prediction", round(float(data_resample(y_pred).values[0]), 2))
                            col2.metric("Actual Mean Weekly Sales", round(float(data_resample(selected_y_test).values[0]), 2))
                            col3.metric('Mean Absolute Error', round(mean_absolute_error(selected_y_test, y_pred), 3))
                            col4.metric('R-squared Score', f'{round(r2_score(selected_y_test, y_pred), 2) * 100}%')

                            st.subheader(f"Prediction Results for Data with No Target in the {interval_string} Period")
                            st.metric("Mean Sales Prediction", round(float(data_resample(y_pred_no_label_df).mean()), 2))

                            plot_target_no_target(y_data, y_pred_df, actual_data_df, y_pred_no_label_df)

if __name__ == '__main__':
    main()
