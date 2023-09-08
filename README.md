# Weekly Sales Forecasting

This project focuses on weekly sales forecasting and includes two Jupyter notebooks for data analysis, preprocessing, feature engineering, and model selection, along with a Streamlit app for interactive forecasting using different models.

## Project Structure

The project is organized as follows:

### Notebooks

1. **Notebook1.ipynb**
   - This notebook contains all the data analysis, preprocessing, and feature engineering steps.
   - Model selection and evaluation are also performed in this notebook.
   - It serves as the foundation for building predictive models.

2. **Notebook2.ipynb**
   - This notebook is dedicated to loading and testing the predictive models developed in Notebook1.
   - Model performance metrics, visualizations, and insights are provided here.

### Product.py

- `Product.py` is a Streamlit application that provides an interactive interface for performing weekly, monthly, and yearly sales forecasting using different models.
- Users can explore and visualize forecasts based on their preferences and input parameters.
- The app leverages the models trained in `Notebook1` and tested in `Notebook2` to provide real-time forecasting results.

### Data Files

- `features.csv`: Contains feature data used for analysis and modeling.
- `stores.csv`: Contains information about different stores.
- `train.csv`: Training data used for model development.
- `test.csv`: Testing data for evaluating model performance.

> The dataset used in this project is sourced from Kaggle. You can find the original dataset [here](https://www.kaggle.com/datasets/yasserh/walmart-dataset).

### Saved Models

- `CatBoostRegressor_model.joblib`: A trained CatBoostRegressor model saved for future use.
- `RandomForest_model.joblib`: A trained RandomForestRegressor model saved for future use.
- `Stacked_model.joblib`: A stacked ensemble model saved for future use.
- `XGBRegressor_model.joblib`: A trained XGBoostRegressor model saved for future use.

## Getting Started

To get started with this project, follow these steps:


1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/hamzahjabari98/Weekly_Sales_Forecasting.git

2. Install the required dependencies.

   ```bash```
   pip install -r requirements.txt

3. Install Streamlit.

   ```bash```
   pip install streamlit

4. Run the Streamlit app

   ```bash```
   streamlit run https://raw.githubusercontent.com/hamzahjabari98/Weekly_Sales_Forecasting/main/Product.py

