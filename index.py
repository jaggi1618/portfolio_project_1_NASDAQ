import streamlit as st
import pandas as pd 
import pickle 
from prophet import Prophet
import datetime
import lightgbm as lgb
import joblib
import plotly.express as px
import plotly.graph_objects as go

data = pd.read_csv('final_data.csv')
data['ds'] = pd.to_datetime(data['ds'])


with open('forecast_model.pkl','rb') as file:
    model_fo = pickle.load(file)

model_lgbm = joblib.load("lgb_model.joblib")   

st.title("NASDAQ-100 (INDEXNASDAQ : NDX) ")
st.subheader("forecasting using Hybrid Model: Prophet + LGBM 🎯 ")

st.subheader('History of Nasdaq-100 index 📈📉')


fig = px.line(data, x='ds', y='y', title="Historical Values", labels={'ds': 'Date', 'y': 'past values of (close)'})
st.plotly_chart(fig)



st.text("""About the source
Nasdaq
Financial services corporation
Nasdaq, Inc. is an American multinational financial services corporation that owns and operates three stock exchanges in the United States: the namesake Nasdaq stock exchange, the Philadelphia Stock ... From Wikipedia
Founded: 8 February 1971
CEO: Adena Friedman (1 Jan 2017–)
Headquarters: New York, New York, United States
Founders: Gordon Macklin""")


start_date = st.date_input("Select Start Date", datetime.date.today())
days_to_predict = st.slider("Select Number of Days to Predict", 1, 365, 30)

st.text('')
st.title('Tuned Prophet based Output')

if "prophet_forecast" not in st.session_state:
    st.session_state.prophet_forecast = None
if "hybrid_forecast" not in st.session_state:
    st.session_state.hybrid_forecast = None
if "forecast" not in st.session_state:
    st.session_state.forecast = None

# Create future dataframe with regressors
future = model_fo.make_future_dataframe(periods=1000)  
future = future.merge(data[['ds', 'volume', 'vix', 'interest_rate', 'cpi']], on='ds', how='left')
future.ffill(inplace=True)  

X_train = future[['volume', 'vix', 'interest_rate', 'cpi']]
# Prophet Model Prediction
if st.button("Predict-Using tuned prophet model"):
    forecast = model_fo.predict(future)
    st.session_state.forecast = forecast  # Store forecast in session state

    # Filter forecast based on user input
    l = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & 
                 (forecast['ds'] < pd.to_datetime(start_date) + pd.Timedelta(days=days_to_predict))
                ].reset_index(drop=True)

    st.session_state.prophet_forecast = l

# Hybrid Model Prediction
if st.button("Predict-Using tuned hybrid model"):
    if st.session_state.forecast is None:
        st.error("Run the Prophet model prediction first!")
    else:
        st.title('Tuned Prophet + LGBM based Output')

        # Use the stored forecast
        forecast = st.session_state.forecast  

        # Ensure we have external regressor values for prediction
        X_train = future[['volume', 'vix', 'interest_rate', 'cpi']]

        # Hybrid model correction
        forecast["lgb_correction"] = model_lgbm.predict(X_train)
        forecast["final_forecast"] = forecast["yhat"] + forecast["lgb_correction"]

        # Filter forecast for the selected range
        k = forecast[(forecast['ds'] >= pd.to_datetime(start_date)) & 
                     (forecast['ds'] < pd.to_datetime(start_date) + pd.Timedelta(days=days_to_predict))
                    ].reset_index(drop=True)

        st.session_state.hybrid_forecast = k

# Display Prophet Forecast
if st.session_state.prophet_forecast is not None:
    st.subheader("Prophet Model Forecast")
    st.dataframe(st.session_state.prophet_forecast)

    fig = px.line(st.session_state.prophet_forecast, x="ds", y="yhat", 
                  title="Forecasted Values (Prophet)", labels={'ds': 'Date', 'yhat': 'Forecasted values'})
    st.plotly_chart(fig)

# Display Hybrid Model Forecast
if st.session_state.hybrid_forecast is not None:
    st.subheader("Hybrid Model Forecast (Prophet + LGBM)")
    st.dataframe(st.session_state.hybrid_forecast)

    fig_1 = px.line(st.session_state.hybrid_forecast, x="ds", y="final_forecast",
                    title="Forecasted Values (Hybrid)", labels={'ds': 'Date', 'final_forecast': 'Forecasted values'})
    st.plotly_chart(fig_1)

    # Comparison Chart
    st.subheader("Comparison Chart: Prophet vs Hybrid Model Forecast")
    fig = go.Figure()

    # Prophet Forecast
    fig.add_trace(go.Scatter(
        x=st.session_state.prophet_forecast["ds"],
        y=st.session_state.prophet_forecast["yhat"],
        mode='lines',
        name='Prophet Forecast'
    ))

    # Hybrid Model Forecast
    fig.add_trace(go.Scatter(
        x=st.session_state.hybrid_forecast["ds"],
        y=st.session_state.hybrid_forecast["final_forecast"],
        mode='lines',
        name='Hybrid Model Forecast'
    ))

    # Layout
    fig.update_layout(
        title="Comparison: Prophet vs. Hybrid Model Forecast",
        xaxis_title="Date",
        yaxis_title="Forecasted Values",
        legend_title="Model"
    )

    st.plotly_chart(fig)


st.subheader('Project Summary: Prophet + LightGBM Hybrid Forecasting Model')

st.text('''

Data Sources:
Historical stock market data (CSV file).
Macroeconomic indicators (Volume, VIX, Interest Rate, CPI) retrieved using the FRED API.
Modeling Approach:

Prophet Model:
Logarithmic growth with multiplicative seasonality.
Tuned changepoint, seasonality, and holidays prior scales.
Integrated external regressors: trading volume, volatility (VIX), interest rate, and CPI.

LightGBM Correction Model:
Trained on residuals (y - Prophet forecast).
Applied as a correction to Prophet's predictions.

Hyperparameter Tuning:
Grid search over changepoint_prior_scale, seasonality_prior_scale, and seasonality_mode.
Best RMSE found using cross-validation.

Evaluation Metrics:
RMSE (Prophet-only Model): 70.11%
RMSE (Hybrid Prophet + LightGBM): 89.12% (Improvement: ~20%)

Deployment:
Model trained in Google Colab and saved using Pickle.
Integrated with Streamlit UI for interactive forecasting.

''')    
