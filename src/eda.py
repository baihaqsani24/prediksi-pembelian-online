import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Folder tempat file ini berada, agar file dataset tetap terbaca di lokal maupun di Hugging Face
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run():
    # Membuat Title
    st.title('Prediksi Potensi Pembelian Sesi Kunjungan Toko Online')

    # Membuat Subheader
    st.subheader('EDA untuk analisis dataset Online Shoppers Purchasing Intention')
    st.write('Page ini dibuat oleh *Baihaqsani* (FTDS-045)')

    # Latar belakang
    st.markdown('---')
    st.write('### Latar Belakang')
    st.write(
        'Mayoritas pengunjung yang membuka toko online tidak jadi membeli. Pada dataset ini, '
        '**84,5% dari 12.330 sesi kunjungan** berakhir tanpa pembelian. Aplikasi ini memprediksi '
        'apakah sebuah sesi kunjungan berpotensi berakhir dengan pembelian berdasarkan perilaku navigasi '
        'pengunjung, sehingga tim marketing dapat menentukan sesi mana yang perlu diprioritaskan dalam pemberian promo.'
    )
    st.write('Sumber data: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468)')

    # Menampilkan DataFrame
    df = pd.read_csv(os.path.join(BASE_DIR, 'online_shoppers_intention.csv'))
    df['Month'] = df['Month'].replace('June', 'Jun')
    df['Revenue'] = df['Revenue'].astype(int)
    st.write('### Dataset')
    st.dataframe(df.head(10))

    # Ringkasan angka
    col1, col2, col3 = st.columns(3)
    col1.metric('Jumlah Sesi', f'{len(df):,}'.replace(',', '.'))
    col2.metric('Sesi Membeli', f"{df['Revenue'].sum():,}".replace(',', '.'))
    col3.metric('Conversion Rate', f"{df['Revenue'].mean() * 100:.1f}%")

    # 1. Distribusi target
    st.markdown('---')
    st.write('### 1. Distribusi Target')
    fig = plt.figure(figsize=(8, 4))
    sns.countplot(x='Revenue', data=df)
    plt.title('Distribusi Revenue (0 = Tidak Membeli, 1 = Membeli)')
    st.pyplot(fig)
    st.write('Target imbalanced: hanya **15,5%** sesi yang berakhir dengan pembelian.')

    # 2. Perilaku navigasi berdasarkan input user
    st.markdown('---')
    st.write('### 2. Perilaku Navigasi Sesi Membeli vs Tidak Membeli')
    opsi = st.selectbox('Pilih Fitur', ('ProductRelated', 'ProductRelated_Duration', 'ExitRates', 'BounceRates',
                                        'Administrative', 'Administrative_Duration'))
    fig = plt.figure(figsize=(8, 4))
    sns.boxplot(x='Revenue', y=opsi, data=df, showfliers=False)
    plt.title(f'{opsi} per kelas target')
    st.pyplot(fig)
    median_table = df.groupby('Revenue')[opsi].median().round(3)
    st.write(f'Median {opsi}: tidak membeli = **{median_table[0]}**, membeli = **{median_table[1]}**')
    st.write('Pembeli membuka ±2x lebih banyak halaman produk, menghabiskan ±2,2x lebih lama di halaman produk, '
             'dan memiliki exit rate lebih rendah.')

    # 3. PageValues
    st.markdown('---')
    st.write('### 3. Conversion Rate berdasarkan PageValues')
    df['PageValues_group'] = np.where(df['PageValues'] > 0, 'PageValues > 0', 'PageValues = 0')
    pv_rate = df.groupby('PageValues_group')['Revenue'].mean() * 100
    fig = plt.figure(figsize=(8, 4))
    sns.barplot(x=pv_rate.index, y=pv_rate.values)
    plt.ylabel('Conversion Rate (%)')
    plt.xlabel('')
    st.pyplot(fig)
    st.write('Sesi yang mengunjungi halaman bernilai memiliki conversion **56,3%**, sedangkan sesi tanpa halaman bernilai hanya **3,9%**.')

    # 4. Tren bulanan
    st.markdown('---')
    st.write('### 4. Jumlah Sesi dan Conversion Rate per Bulan')
    month_order = ['Feb', 'Mar', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_summary = df.groupby('Month')['Revenue'].agg(['count', 'mean']).reindex(month_order)
    fig, ax1 = plt.subplots(figsize=(12, 4))
    sns.barplot(x=month_summary.index, y=month_summary['count'], color='lightsteelblue', ax=ax1)
    ax1.set_ylabel('Jumlah Sesi')
    ax2 = ax1.twinx()
    ax2.plot(range(len(month_summary)), month_summary['mean'] * 100, color='darkred', marker='o')
    ax2.set_ylabel('Conversion Rate (%)')
    st.pyplot(fig)
    st.write('November memiliki conversion tertinggi (**25,4%**). Data Januari dan April tidak tersedia di dataset.')

    # 5. Conversion rate kolom kategorikal berdasarkan input user
    st.markdown('---')
    st.write('### 5. Conversion Rate per Kategori')
    opsi_cat = st.selectbox('Pilih Kolom Kategorikal', ('VisitorType', 'TrafficType', 'SpecialDay', 'Weekend', 'OperatingSystems', 'Region'))
    cat_summary = df.groupby(opsi_cat)['Revenue'].agg(['count', 'mean']).reset_index()
    cat_summary.columns = [opsi_cat, 'jumlah sesi', 'conversion rate (%)']
    cat_summary['conversion rate (%)'] = (cat_summary['conversion rate (%)'] * 100).round(1)
    cat_summary[opsi_cat] = cat_summary[opsi_cat].astype(str)
    fig = px.bar(cat_summary, x=opsi_cat, y='conversion rate (%)', hover_data=['jumlah sesi'])
    st.plotly_chart(fig)

    # 6. Plotly scatter
    st.markdown('---')
    st.write('### 6. Jumlah vs Durasi Halaman Produk')
    df['Revenue_label'] = df['Revenue'].map({0: 'Tidak Membeli', 1: 'Membeli'})
    fig = px.scatter(df, x='ProductRelated', y='ProductRelated_Duration', color='Revenue_label',
                     hover_data=['PageValues', 'Month', 'VisitorType'], opacity=0.5)
    st.plotly_chart(fig)


if __name__ == '__main__':
    run()
