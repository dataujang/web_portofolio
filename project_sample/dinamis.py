import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(layout="wide")

st.title("Simulasi Tarif Diskon KA Bandara")

# Data dasar
trains_number = {
    'KA MRI_BST':[801802,805806,809810,813814,817818,821822,825826,829830,833834,841842,845846,849850,853854,861862,
                  865866,869870,873874,877878,881882,885886,889890,893894,897898,901902,905906,909910,913914,917918,
                  921922,925926,929930,933934],
    'KA BST_MRI':[803804,807808,811812,815816,819820,823824,827828,831832,835836,843844,847848,851852,855856,863864,
                  867868,871872,875876,879880,883884,887888,890891,895896,899890,903904,907908,911912,915916,919920,
                  923924,927928,931932,935936]
}

relasi_list = { 
    'Relasi MRI_BST':['MRI-SUDB','MRI-DU','MRI-BPR','MRI-RW','MRI-BST','SUDB-DU','SUDB-BPR','SUDB-RW','SUDB-BST','DU-BPR','DU-RW','DU-BST','RW-BPR','RW-BST','BPR-BST'],
    'Relasi BST_MRI':['BST-BPR','BST-RW','BST-DU','BST-SUDB','BST-MRI','BPR-RW','BPR-DU','BPR-SUDB','BPR-MRI','RW-DU','RW-SUDB','RW-MRI','DU-SUDB','DU-MRI','SUDB-MRI'],
}

# Semua relasi unik
relasi_unik = list(set(relasi_list['Relasi MRI_BST'] + relasi_list['Relasi BST_MRI']))

st.sidebar.header("Input Tarif Dasar per Relasi")

# Form input tarif dasar
tarif_input = {}
default_tarif = {
    'MRI-SUDB':10000,'MRI-DU':10000,'MRI-BPR':35000,'MRI-RW':35000,'MRI-BST':80000,
    'SUDB-DU':10000,'SUDB-BPR':35000,'SUDB-RW':35000,'SUDB-BST':80000,
    'DU-BPR':25000,'DU-RW':25000,'DU-BST':70000,
    'RW-BPR':25000,'RW-BST':35000,'BPR-BST':35000,
    'BST-BPR':35000,'BST-RW':35000,'BST-DU':70000,'BST-SUDB':80000,'BST-MRI':80000,
    'BPR-RW':35000,'BPR-DU':25000,'BPR-SUDB':35000,'BPR-MRI':35000,
    'RW-DU':25000,'RW-SUDB':35000,'RW-MRI':35000,
    'DU-SUDB':10000,'DU-MRI':10000,'SUDB-MRI':10000
}

# Input tarif via sidebar
for relasi in sorted(relasi_unik):
    tarif_input[relasi] = st.sidebar.number_input(f'Tarif Dasar {relasi}', min_value=1000, value=default_tarif.get(relasi,10000), step=1000)

# Proses Data
departure_start_dates = datetime.strptime("20250201", "%Y%m%d")
departure_end_dates = datetime.strptime("20250228", "%Y%m%d")

data = []
current_date = departure_start_dates

while current_date <= departure_end_dates:
    payment_dates = pd.date_range(start=current_date - timedelta(days=7), end=current_date)
    for no_ka in trains_number['KA MRI_BST']:
        for relasi in relasi_list['Relasi MRI_BST']:
            for payment in payment_dates:
                data.append({
                    'No KA': no_ka,
                    'Relasi': relasi,
                    'Payment Date': payment.date(),
                    'Trip Date': current_date.date(),
                })
    for no_ka in trains_number['KA BST_MRI']:
        for relasi in relasi_list['Relasi BST_MRI']:
            for payment in payment_dates:
                data.append({
                    'No KA': no_ka,
                    'Relasi': relasi,
                    'Payment Date': payment.date(),
                    'Trip Date': current_date.date(),
                })
    current_date += timedelta(days=1)

df = pd.DataFrame(data)
df['Payment Date'] = pd.to_datetime(df['Payment Date'])
df['Trip Date'] = pd.to_datetime(df['Trip Date'])
df['Selisih Hari Pembelian'] = (df['Trip Date'] - df['Payment Date']).dt.days

# Tarif Dasar input user
df['Tarif Dasar'] = df['Relasi'].map(tarif_input)

# Diskon Vectorized
diskon_relasi = ['MRI-BST', 'SUDB-BST', 'DU-BST', 'BST-MRI', 'BST-SUDB', 'BST-DU']
mask_diskon = df['Relasi'].isin(diskon_relasi)
diskon_persen = 0.05 * df['Selisih Hari Pembelian']
tarif_sales = df['Tarif Dasar'] * (1 - diskon_persen)
tarif_sales[~mask_diskon] = df.loc[~mask_diskon, 'Tarif Dasar']
tarif_sales[mask_diskon] = np.maximum(tarif_sales[mask_diskon], 50000)
df['Tarif Sales'] = (np.round(tarif_sales / 1000) * 1000).astype(int)

st.subheader("Preview Data Tarif Sales")
st.dataframe(df)

# Download CSV
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    "Download Hasil CSV",
    data=csv,
    file_name='simulasi_tarif_sales.csv',
    mime='text/csv'
)
