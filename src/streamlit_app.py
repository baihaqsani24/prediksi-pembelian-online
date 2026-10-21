import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title = 'Prediksi Pembelian Toko Online',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

# Navigasi halaman
page = st.sidebar.selectbox('Pilih Halaman', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()
else:
    prediction.run()
