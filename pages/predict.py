import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
from datetime import date

import yfinance as yf

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

from copy import deepcopy

#TRAINING MODEL 

from tensorflow.keras.models import load_model
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras import layers

START = '2004-06-08'
TODAY = date.today().strftime("%Y-%m-%d")

def main():

    st.title("Stock Prediction")
    st.write("Analyze stock performance, view predictions, and make informed decisions.")
    st.divider()  # Garis horizontal


    st.write(""
             "")

    stocks=("BBCA.JK")
    # st.markdown(
    #     '<h3 style="color: gold;">How many days before you want to see for comparison?</h3><br></br>',
    #     unsafe_allow_html=True,
    # )
    # selected_stock = st.selectbox("\nSelect dataset for prediction", stocks)

    # n_years = st.slider("Day of prediction:", 1, 4)
    # n_years = st.slider("Day of prediction:", 1, 4, key="temp_slider")
    n_years=3
    # period = n_years *365

    # @st.cache_data
    # def load_data(ticker):
    #     data = yf.download(ticker, START, TODAY)
    #     data.reset_index(inplace=True)
    #     return data
    def load_data():
        data = pd.read_csv("BBCA_data.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    
    # data_load_state = st.text("Load data...")
    data = load_data()
    # data_load_state.text("Loading data...done!")

    # Debug: Periksa kolom data
    # st.write("Data columns:", data.columns)

    # Periksa apakah data memiliki MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
        
    st.subheader("Stock data")
    st.write(data.tail(30))

    def plot_raw_data():
        fig = go.Figure()

        #plot raw data
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'],mode ="lines",name='stock_open'))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'],mode ="lines",name='stock_close'))
        fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
        st.plotly_chart(fig)

    plot_raw_data()

    # st.line_chart(data[['Date', 'Close']].set_index('Date'))



    # Siapkan data untuk Prophet
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    df_train['ds'] = pd.to_datetime(df_train['ds'])
    df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
    df_train = df_train.dropna(subset=['y'])

    # m = Prophet()
    # m.fit(df_train)

    #LOAD MODEL
    model = load_model("model_lstm.h5")

    #ngubah data menjadi windowed df
    data = data[["Date", "Close"]]
    # def str_to_datetime(s):
    #     split = s.split('-')
    #     year, month, day = int(split[0]), int(split[1]), int(split[2])
    #     return datetime.datetime(year=year, month=month, day=day)

    #ngubah data menjadi windowed df
    def df_to_windowed_df(dataframe, first_date_str, last_date_str, n_years):
        first_date = str_to_datetime(first_date_str)
        last_date  = str_to_datetime(last_date_str)

        target_date = first_date
        
        dates = []
        X, Y = [], []

        last_time = False
        while True:
            # print(f"Processing: {target_date}")
            df_subset = dataframe.loc[:target_date].tail(n_years+1)
            
            if len(df_subset) != n_years+1:
                # print(f'Error: Window of size {n} is too large for date {target_date}')
                return

            values = df_subset['Close'].to_numpy()
            x, y = values[:-1], values[-1]

            dates.append(target_date)
            X.append(x)
            Y.append(y)

            next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
            next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
            next_date_str = next_datetime_str.split('T')[0]
            year_month_day = next_date_str.split('-')
            year, month, day = year_month_day
            next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))
            if next_date == target_date:
                if next_date == target_date:
                    print(f"Warning: next_date ({next_date}) sama dengan target_date ({target_date}), mencari tanggal berikutnya...")
            
            # Cari semua tanggal yang lebih besar dari target_date
            future_dates = dataframe.index[dataframe.index > target_date]
            
            if len(future_dates) > 0:
                next_date = future_dates[0]  # Ambil tanggal pertama yang lebih besar
            else:
                # print(f"Tidak ada tanggal berikutnya setelah {target_date}, loop berhenti.")
                break
            
            if last_time:
                break
            
            target_date = next_date

            if target_date >= last_date:
                last_time = True
            
            # print(next_week)
            
        ret_df = pd.DataFrame({})
        ret_df['Target Date'] = dates
        
        X = np.array(X)
        for i in range(0, n_years):
            # X[:, i]
            ret_df[f'Target-{n_years-i}'] = X[:, i]
        
        ret_df['Target'] = Y

        return ret_df
    # Helper function untuk konversi string ke datetime
    def str_to_datetime(date_str):
        """Konversi string ke datetime."""
        year, month, day = map(int, date_str.split('-'))
        return datetime.datetime(year=year, month=month, day=day)

    #n = mau melihat berapa data kebelakang, karena ini LSTM
    # print(data)
    data.index = data.pop('Date')
    start = '2004-07-12'
    end = '2025-02-07'
    
    # windowed_df = df_to_windowed_df(data, start , end, n_years)
    def process_data(data, start, end, n_years):
        return df_to_windowed_df(data, start, end, n_years)

    # Gunakan dalam operasi selanjutnya
    windowed_df = process_data(data, start, end, n_years)
    # with st.expander("View Target Data"):
    #     st.write(windowed_df)
    # print(windowed_df)
    # windowed_df = pd.DataFrame(windowed_df)

    #ngubah bentuk windowed DF menjadi tabel
    # Date | X | Y
    # X = data harga kebelakang sebelum hari tersebut
    # y = harga di hari tersebut
    # date = tanggal
    def windowed_df_to_date_X_y(windowed_dataframe):
        df_as_np = windowed_dataframe.to_numpy()

        dates = df_as_np[:, 0]

        middle_matrix = df_as_np[:, 1:-1]
        X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

        Y = df_as_np[:, -1]

        return dates, X.astype(np.float32), Y.astype(np.float32)

    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    print(dates.shape, X.shape, y.shape)

    #membagi table
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

    ####

    train_predictions = model.predict(X_train).flatten()
    ##
    val_predictions = model.predict(X_val).flatten()

    ##
    test_predictions = model.predict(X_test).flatten()

    #PREDICT
    train_predictions = model.predict(X_train).flatten()

    def plot_predictions(dates_train, train_predictions, y_train,
                        dates_val, val_predictions, y_val,
                        dates_test, test_predictions, y_test): #menghapus recursive_date, recursive_prediction
        # figur baru
        fig = go.Figure()

        # garis Training data
        fig.add_trace(go.Scatter(x=dates_train, y=train_predictions, mode='lines', name='Training Predictions', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=dates_train, y=y_train, mode='lines', name='Training Observations', line=dict(color='cyan')))

        # Validation data
        fig.add_trace(go.Scatter(x=dates_val, y=val_predictions, mode='lines', name='Validation Predictions', line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=dates_val, y=y_val, mode='lines', name='Validation Observations', line=dict(color='gold')))

        # garis testing
        fig.add_trace(go.Scatter(x=dates_test, y=test_predictions, mode='lines', name='Testing Predictions', line=dict(color='red')))
        fig.add_trace(go.Scatter(x=dates_test, y=y_test, mode='lines', name='Testing Observations', line=dict(color='pink')))

       
        # layout dengan range slider
        fig.update_layout(
            title_text="Train, Validation, Test Predictions",
            xaxis=dict(
                title="Dates",
                rangeslider=dict(visible=True),  # range slider
                rangeselector=dict(             # milih range
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")  # Menampilkan semua data
                    ])
                )
            ),
            yaxis=dict(
                title="Values",
            ),
            template="plotly_dark"
        )

        # fig.show()
        st.plotly_chart(fig)

    plot_predictions(dates_train, train_predictions, y_train,
                        dates_val, val_predictions, y_val,
                        dates_test, test_predictions, y_test)

    def plot_future_predictions(day_pred, test_res):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=day_pred,
            y=test_res,
            mode='lines+markers',
            name='Future Prediction',
            marker=dict(color='gold'),
            line=dict(color='gold')
        ))

        # minimum warna merah
        min_index = test_res.index(min(test_res))
        fig.add_trace(go.Scatter(
            x=[day_pred[min_index]],
            y=[test_res[min_index]],
            mode='markers+text',
            marker=dict(color='red', size=10),
            name='Min Value',
            text=[f"Min: {min(test_res):.2f}"],
            textposition="top center"
        ))

        # max hijau
        max_index = test_res.index(max(test_res))
        fig.add_trace(go.Scatter(
            x=[day_pred[max_index]],
            y=[test_res[max_index]],
            mode='markers+text',
            marker=dict(color='green', size=10),
            name='Max Value',
            text=[f"Min: {max(test_res):.2f}"],
            textposition="top center"
        ))

        fig.update_layout(
            title="Future Predictions",
            xaxis_title="Date",
            yaxis_title="Predicted Close Price",
            template="plotly_dark" 
        )
        st.plotly_chart(fig)

    # st.subheader("")
    st.divider()  # Garis horizontal
    st.markdown(
        '<h2 style="color: gold;">Future Prediction (Forecast)</h2><br></br>',
        unsafe_allow_html=True,
    )

    st.number_input("How many days do you want to predict?", min_value=1, max_value=100, key="temp_day")

    if st.button("Prediksi ke Depan"):
        st.session_state["day"] = st.session_state["temp_day"]

    if "day" in st.session_state:
        X_last = X[-1]
        X_last = X_last.reshape(1, X_last.shape[0], X_last.shape[1])
        test_res = []
        curr_date = pd.to_datetime(dates[-1])
        day_pred = []

        for i in range(st.session_state["day"]):
            test_predictions = model.predict(X_last).flatten()
            X_last = np.roll(X_last, -1, axis=1)
            X_last[0, -1, 0] = test_predictions[0]
            now_date = curr_date + datetime.timedelta(days=1)
            curr_date = now_date
            day_pred.append(now_date)
            test_res.append(test_predictions[0])

        st.markdown(
            f"""
            <p>Maximum prediction: <span style="color:green;">{max(test_res):.2f}</span> on {day_pred[test_res.index(max(test_res))]}</p>
            <p>Minimum prediction: <span style="color:red;">{min(test_res):.2f}</span> on {day_pred[test_res.index(min(test_res))]}</p>
            """,
            unsafe_allow_html=True
        )
        # st.write("Maximum prediction:", max(test_res), "on", day_pred[test_res.index(max(test_res))])
        # st.write("Minimum prediction:", min(test_res), "on", day_pred[test_res.index(min(test_res))])


        plot_future_predictions(day_pred, test_res)

main()