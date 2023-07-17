import pandas as pd
import numpy as np

class DataLoadMergePreprocess:
    def __init__(self, features_path='features.csv', stores_path='stores.csv', train_path='train.csv'):
        self.features_path = features_path
        self.stores_path = stores_path
        self.train_path = train_path
        self.df = None

    def load_merge_data(self):
        """
        Load and merge data from features, stores, and train CSV files into one dataframe.
        """
        # Load data from CSV files
        self.features = pd.read_csv(self.features_path)
        self.stores = pd.read_csv(self.stores_path)
        self.train = pd.read_csv(self.train_path)

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
        self.df = pd.merge(train_stores, self.features, on=[
                           'Store', 'Date', 'IsHoliday'])

    def preprocess(self):
        if self.df is None:
            raise Exception("Data has not been loaded and merged yet.")
        
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Year_quarter'] = self.df['Date'].dt.quarter
        self.df['Day'] = self.df['Date'].dt.day
        self.df['Week'] = self.df['Date'].dt.isocalendar().week
        self.df['Month'] = self.df['Date'].dt.month
        self.df['Year'] = self.df['Date'].dt.year
        self.df['Temperature'] = (self.df['Temperature'] - 32) * 5/9
        self.df = self.add_holiday_columns(self.df)


    def add_holiday_columns(self, df):
        Super_Bowl = pd.to_datetime(["2010-02-12", "2011--02-11", "2012-02-10"])
        Labour_Day = pd.to_datetime(["2010-09-10", "2011-09-09", "2012-09-07"])
        Thanksgiving_Day = pd.to_datetime(["2010-11-26", "2011-11-25"])
        Christmas = pd.to_datetime(["2010-12-31", "2011-12-30"])

        df["Super_Bowl"] = df["Date"].isin(Super_Bowl)
        df["Labour_Day"] = df["Date"].isin(Labour_Day)
        df["Thanksgiving_Day"] = df["Date"].isin(Thanksgiving_Day)
        df["Christmas"] = df["Date"].isin(Christmas)
        return df
