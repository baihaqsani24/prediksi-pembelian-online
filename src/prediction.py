# Import libraries
import os
import pickle
import json
import pandas as pd
import numpy as np
import streamlit as st

# Folder tempat file ini berada, agar file model tetap terbaca di lokal maupun di Hugging Face
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all files
with open(os.path.join(BASE_DIR, 'model_rf_tuned.pkl'), 'rb') as file_1:
    model = pickle.load(file_1)

with open(os.path.join(BASE_DIR, 'list_top_categories.txt'), 'r') as file_2:
    list_top_categories = json.load(file_2)


def preprocess(data_inf):
    # Proses yang sama seperti saat training (di luar pipeline)
    data_inf = data_inf.copy()

    # 1. Penyesuaian data
    data_inf['Month'] = data_inf['Month'].replace('June', 'Jun')
    code_cols = ['OperatingSystems', 'Browser', 'Region', 'TrafficType']
    data_inf[code_cols] = data_inf[code_cols].astype(str)
    data_inf['Weekend'] = data_inf['Weekend'].astype(int)

    # 2. Handling cardinality
    for col in ['OperatingSystems', 'TrafficType']:
        data_inf[col] = np.where(data_inf[col].isin(list_top_categories[col]), data_inf[col], 'Other')

    # 3. Drop kolom hasil feature selection
    return data_inf.drop(['Region', 'Browser', 'BounceRates'], axis=1)


def run():
    st.title('Prediksi Potensi Pembelian')
    st.write('Masukkan data aktivitas satu sesi kunjungan, lalu klik **Predict** untuk melihat apakah sesi tersebut berpotensi berakhir dengan pembelian.')

    with st.form(key='form_online_shoppers'):
        st.write('#### Aktivitas Halaman')
        col1, col2 = st.columns(2)
        administrative = col1.number_input('Administrative (jumlah halaman akun/login)', min_value=0, max_value=100, value=2)
        administrative_duration = col2.number_input('Administrative Duration (detik)', min_value=0.0, max_value=10000.0, value=60.0)
        informational = col1.number_input('Informational (jumlah halaman informasi)', min_value=0, max_value=100, value=0)
        informational_duration = col2.number_input('Informational Duration (detik)', min_value=0.0, max_value=10000.0, value=0.0)
        product_related = col1.number_input('Product Related (jumlah halaman produk)', min_value=0, max_value=1000, value=20)
        product_related_duration = col2.number_input('Product Related Duration (detik)', min_value=0.0, max_value=100000.0, value=600.0)
        st.markdown('---')

        st.write('#### Metrik Google Analytics')
        col3, col4, col5 = st.columns(3)
        bounce_rates = col3.number_input('Bounce Rates', min_value=0.0, max_value=0.2, value=0.003, step=0.001, format='%.3f', help='Rentang 0 - 0.2')
        exit_rates = col4.number_input('Exit Rates', min_value=0.0, max_value=0.2, value=0.025, step=0.001, format='%.3f', help='Rentang 0 - 0.2')
        page_values = col5.number_input('Page Values', min_value=0.0, max_value=500.0, value=0.0, help='Rata-rata nilai halaman yang dikunjungi sebelum transaksi')
        st.markdown('---')

        st.write('#### Waktu & Pengunjung')
        col6, col7 = st.columns(2)
        month = col6.selectbox('Month', ('Feb', 'Mar', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), index=8,
                               help='Januari dan April tidak tersedia di data training')
        special_day = col7.selectbox('Special Day', (0.0, 0.2, 0.4, 0.6, 0.8, 1.0), index=0,
                                     help='Kedekatan dengan hari spesial (1 = paling dekat)')
        visitor_type = col6.selectbox('Visitor Type', ('Returning_Visitor', 'New_Visitor', 'Other'), index=0)
        weekend = col7.selectbox('Weekend', ('Tidak', 'Ya'), index=0)
        st.markdown('---')

        st.write('#### Informasi Teknis (kode dari Google Analytics)')
        col8, col9, col10, col11 = st.columns(4)
        operating_systems = col8.selectbox('Operating Systems', list(range(1, 9)), index=1)
        browser = col9.selectbox('Browser', list(range(1, 14)), index=1)
        region = col10.selectbox('Region', list(range(1, 10)), index=0)
        traffic_type = col11.selectbox('Traffic Type', list(range(1, 21)), index=1)

        submitted = st.form_submit_button('Predict')

    # Create a new data (semua kolom asli, bukan hanya hasil feature selection)
    data_inf = {
        'Administrative': administrative,
        'Administrative_Duration': administrative_duration,
        'Informational': informational,
        'Informational_Duration': informational_duration,
        'ProductRelated': product_related,
        'ProductRelated_Duration': product_related_duration,
        'BounceRates': bounce_rates,
        'ExitRates': exit_rates,
        'PageValues': page_values,
        'SpecialDay': special_day,
        'Month': month,
        'OperatingSystems': operating_systems,
        'Browser': browser,
        'Region': region,
        'TrafficType': traffic_type,
        'VisitorType': visitor_type,
        'Weekend': weekend == 'Ya'
    }
    data_inf = pd.DataFrame([data_inf])
    st.write('#### Data yang akan diprediksi')
    st.dataframe(data_inf)

    if submitted:
        data_inf_final = preprocess(data_inf)

        # Predict using Random Forest (pipeline)
        y_pred_inf = model.predict(data_inf_final)[0]
        proba_inf = model.predict_proba(data_inf_final)[0, 1]

        st.write('## Hasil Prediksi')
        st.write(f'Probabilitas membeli: **{proba_inf:.1%}**')
        st.progress(float(proba_inf))

        if y_pred_inf == 1:
            st.success('Sesi ini **berpotensi membeli**. Tidak perlu promo tambahan, pastikan proses checkout lancar '
                       '(misalnya tampilkan rekomendasi produk pelengkap).')
        elif proba_inf >= 0.3:
            st.warning('Sesi ini **belum diprediksi membeli**, tetapi peluangnya cukup besar. '
                       'Layak diberi pendekatan ringan seperti voucher terbatas atau pengingat produk yang dilihat.')
        else:
            st.error('Sesi ini **tidak berpotensi membeli**, sehingga tidak menjadi prioritas pemberian promo.')


if __name__ == '__main__':
    run()
