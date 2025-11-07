# Stock Prediction
###
**Link website (deployed)**: https://stockprediction-machine-learning123.streamlit.app/

comparing machine learning models : LSTM and Linear Regression to predict BBCA stock prices using historical data. Demonstrates how AI can enhance financial forecasting by capturing both linear trends and complex temporal patterns.

- High-Performance LSTM Model: Achieved R² Score of 0.9982 demonstrating ability to capture complex temporal patterns in stock price data.
- Baseline Linear Regression: Delivered R² Score of 0.9928 offering a transparent and interpretable benchmark for comparison.
- Time-Series Forecasting: Applied advanced sequence modeling using LSTM to handle long-term dependencies and non-linear trends in financial data.
- Model Evaluation & Error Analysis: Compared models using MAE, MSE, and R² metrics to assess precision, variance explanation, and sensitivity to outliers.

## 🛠️ Tech Stack
- Built with Python : Pandas, NumPy, Prophet, Scikit-learn, TensorFlow/Keras, and visualized insights using Matplotlib, and Plotly

## Run Streamlit App - Local

1. **Clone atau Unduh Proyek dari GitHub**

   Clone repository:
    ```bash
    git clone https://github.com/gracebirgita/stockPrediction.git
    ```
 2. **Go to directory**
    ```bash
    cd stockPrediction
    ```
 3. **Install dependencies**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/Mac
    venv\Scripts\activate     # Windows
    ```

    Run command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```
 4. **Run Streamlit App**

    ```bash
    streamlit run nav.py
    ```
