# Qualité & Livraison

# Library import
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Page config
st.set_page_config(page_title='Qualité & Livraison', layout='wide')

# Data loading
if 'data' not in st.session_state:
    st.session_state['data'] = pd.read_csv("dataset_industry.csv")
data = st.session_state['data']
data['order_date'] = pd.to_datetime(data['order_date'])

# Sidebar filters
st.sidebar.header("Filtres")
start = st.sidebar.date_input("Date de début", value=data['order_date'].min())
end = st.sidebar.date_input("Date de fin", value=data['order_date'].max())
selected_families = st.sidebar.multiselect("Famille de produits", options=data['product_family'].unique(), default=data['product_family'].unique())
selected_clients = st.sidebar.multiselect("Client", options=data['client_name'].unique(), default=data['client_name'].unique())

# Filtered data
filtered_data = data[
    (data['order_date'] >= pd.Timestamp(start)) &
    (data['order_date'] <= pd.Timestamp(end)) &
    (data['product_family'].isin(selected_families)) &
    (data['client_name'].isin(selected_clients))
]

# Calculations
otd_rate = (filtered_data['on_time'] == 'Yes').mean() * 100
non_conformity_rate = (filtered_data['non_conformity'] == 'Yes').mean() * 100
total_orders = len(filtered_data)
late_orders = (filtered_data['on_time'] == 'No').sum()
non_conform_orders = (filtered_data['non_conformity'] == 'Yes').sum()

otd_by_family = filtered_data.groupby('product_family').apply(lambda x: (x['on_time'] == 'Yes').mean() * 100).reset_index()
otd_by_family.columns = ['product_family', 'otd_rate']

nc_by_family = filtered_data.groupby('product_family').apply(lambda x: (x['non_conformity'] == 'Yes').mean() * 100).reset_index()
nc_by_family.columns = ['product_family', 'nc_rate']

otd_by_client = filtered_data.groupby('client_name').apply(lambda x: (x['on_time'] == 'Yes').mean() * 100).reset_index()
otd_by_client.columns = ['client_name', 'otd_rate']
otd_by_client = otd_by_client.sort_values('otd_rate', ascending=True)

otd_by_month = filtered_data.groupby(filtered_data['order_date'].dt.to_period('M')).apply(lambda x: (x['on_time'] == 'Yes').mean() * 100).reset_index()
otd_by_month.columns = ['order_date', 'otd_rate']
otd_by_month['order_date'] = otd_by_month['order_date'].astype(str)

# Calcul période précédente
period_duration = (pd.Timestamp(end) - pd.Timestamp(start)).days
prev_start = pd.Timestamp(start) - pd.Timedelta(days=period_duration)
prev_end = pd.Timestamp(start) - pd.Timedelta(days=1)

prev_data = data[
    (data['order_date'] >= prev_start) &
    (data['order_date'] <= prev_end) &
    (data['product_family'].isin(selected_families)) &
    (data['client_name'].isin(selected_clients))
]

prev_otd = (prev_data['on_time'] == 'Yes').mean() * 100 if len(prev_data) > 0 else 0
prev_nc = (prev_data['non_conformity'] == 'Yes').mean() * 100 if len(prev_data) > 0 else 0
prev_late = (prev_data['on_time'] == 'No').sum() if len(prev_data) > 0 else 0
prev_non_conform = (prev_data['non_conformity'] == 'Yes').sum() if len(prev_data) > 0 else 0

delta_otd = otd_rate - prev_otd
delta_nc = non_conformity_rate - prev_nc
delta_late = late_orders - prev_late
delta_non_conform = non_conform_orders - prev_non_conform

# System prompt
system_prompt = f"""Tu es un analyste qualité s'adressant à la direction d'une TPE/PME. Ton ton est professionnel et orienté action.

Voici les données qualité et livraison :
- Taux OTD global : {otd_rate:.1f}% (objectif : 95%)
- Taux de non-conformité : {non_conformity_rate:.1f}%
- Nombre de commandes en retard : {late_orders}
- Nombre de non-conformités : {non_conform_orders}
- OTD par famille de produits : {otd_by_family.to_string()}
- Non-conformités par famille : {nc_by_family.to_string()}
- OTD par client : {otd_by_client.to_string()}

Ton analyse doit :
- Identifier les familles de produits ou clients avec les taux OTD les plus faibles
- Alerter sur les dérives de non-conformité par famille ou client
- Proposer des pistes concrètes pour réduire les retards et les rebuts
- Quantifier l'impact financier potentiel des non-conformités si possible

Réponds de manière concise, structurée et avec des recommandations actionnables."""

# AI Chat
st.sidebar.markdown("---")
st.sidebar.subheader("Analyse IA")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_question = st.sidebar.text_input("Posez une question...")

if user_question:
    st.session_state['messages'].append({"role": "user", "content": user_question})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system_prompt,
        messages=st.session_state['messages']
    )
    st.session_state['messages'].append({"role": "assistant", "content": response.content[0].text})

for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.sidebar.write("**Vous :** " + msg['content'])
    else:
        st.sidebar.write("**IA :** " + msg['content'])

# Page title
st.title('Qualité & Livraison')

# Global KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Taux OTD", f"{otd_rate:.1f}%", delta=f"{delta_otd:.1f}%")
col2.metric("Taux non-conformité", f"{non_conformity_rate:.1f}%", delta=f"{delta_nc:.1f}%", delta_color="inverse")
col3.metric("Commandes en retard", f"{late_orders:,}", delta=f"{delta_late:+.0f}", delta_color="inverse")
col4.metric("Non-conformités", f"{non_conform_orders:,}", delta=f"{delta_non_conform:+.0f}", delta_color="inverse")

# OTD by month
st.markdown("---")
st.subheader("Taux OTD par mois")
fig = px.line(otd_by_month, x='order_date', y='otd_rate', color_discrete_sequence=["#4a5568"])
fig.update_traces(hovertemplate='%{x}<br>%{y:.1f}%')
fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Objectif 95%")
st.plotly_chart(fig, use_container_width=True)

# OTD & NC by family
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("OTD par famille de produits")
    fig = px.bar(otd_by_family, x='product_family', y='otd_rate', color_discrete_sequence=["#4a5568"])
    fig.update_traces(hovertemplate='%{x}<br>%{y:.1f}%')
    fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Objectif 95%")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Taux de non-conformité par famille")
    fig = px.bar(nc_by_family, x='product_family', y='nc_rate', color_discrete_sequence=["#4a5568"])
    fig.update_traces(hovertemplate='%{x}<br>%{y:.1f}%')
    st.plotly_chart(fig, use_container_width=True)

# OTD by client
st.markdown("---")
st.subheader("OTD par client")
fig = px.bar(otd_by_client, x='client_name', y='otd_rate', color_discrete_sequence=["#4a5568"])
fig.update_traces(hovertemplate='%{x}<br>%{y:.1f}%')
fig.update_layout(xaxis_tickangle=-45)
fig.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Objectif 95%")
st.plotly_chart(fig, use_container_width=True)