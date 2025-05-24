import streamlit as st

dashboard_page = st.Page(
    "pages/dashboard.py", title="Dashboard", icon=":material/search:"
)

main_page = st.Page(
    "pages/predict.py", title="Stock Prediction", icon=":material/dashboard:"
)
    
pg = st.navigation([dashboard_page, main_page])

pg.run()