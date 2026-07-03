# Analyse industrielle - Chiffres globaux

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
st.set_page_config(page_title='Analyse Industrielle', layout='wide')

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
total_revenue = filtered_data['revenue'].sum()
total_orders = len(filtered_data)
otd_rate = (filtered_data['on_time'] == 'Yes').mean() * 100
non_conformity_rate = (filtered_data['non_conformity'] == 'Yes').mean() * 100

revenue_by_family = filtered_data.groupby('product_family')['revenue'].sum().reset_index()
revenue_by_client = filtered_data.groupby('client_name')['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(10)
revenue_by_month = filtered_data.groupby(filtered_data['order_date'].dt.to_period('M'))['revenue'].sum().reset_index()
revenue_by_month['order_date'] = revenue_by_month['order_date'].astype(str)

# Calculation compare period before
period_duration = (pd.Timestamp(end) - pd.Timestamp(start)).days
prev_start = pd.Timestamp(start) - pd.Timedelta(days=period_duration)
prev_end = pd.Timestamp(start) - pd.Timedelta(days=1)

prev_data = data[
    (data['order_date'] >= prev_start) &
    (data['order_date'] <= prev_end) &
    (data['product_family'].isin(selected_families)) &
    (data['client_name'].isin(selected_clients))
]

prev_revenue = prev_data['revenue'].sum()
prev_orders = len(prev_data)
prev_otd = (prev_data['on_time'] == 'Yes').mean() * 100 if len(prev_data) > 0 else 0
prev_nc = (prev_data['non_conformity'] == 'Yes').mean() * 100 if len(prev_data) > 0 else 0

delta_revenue = total_revenue - prev_revenue
delta_orders = total_orders - prev_orders
delta_otd = otd_rate - prev_otd
delta_nc = non_conformity_rate - prev_nc

system_prompt = f"""Tu es un analyste de données industrielles s'adressant à la direction d'une TPE/PME. Ton ton est professionnel et orienté action.

Voici les données de l'entreprise :
- Chiffre d'affaires total : {total_revenue:,.2f} €
- Nombre de commandes : {total_orders}
- Taux de livraison à l'heure (OTD) : {otd_rate:.1f}%
- Taux de non-conformité : {non_conformity_rate:.1f}%
- CA par famille de produits : {revenue_by_family.to_string()}
- Top 10 clients : {revenue_by_client.to_string()}
- CA par mois : {revenue_by_month.to_string()}

Ton analyse doit :
- Identifier la tendance du CA sur la période
- Identifier les familles de produits les plus performantes
- Repérer les mois creux et les pics d'activité
- Proposer des pistes concrètes pour augmenter le CA (mix produit, saisonnalité, pricing)

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
st.title('Analyse Industrielle')

# Global KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Chiffre d'affaires", f"{total_revenue:,.2f} €", delta=f"{delta_revenue:,.2f} €")
col2.metric("Nombre de commandes", f"{total_orders:,}", delta=f"{delta_orders:,}")
col3.metric("Taux OTD", f"{otd_rate:.1f}%", delta=f"{delta_otd:.1f}%")
col4.metric("Taux non-conformité", f"{non_conformity_rate:.1f}%", delta=f"{delta_nc:.1f}%", delta_color="inverse")

# Revenue by month
st.markdown("---")
st.subheader("CA par mois")
fig = px.line(revenue_by_month, x='order_date', y='revenue', color_discrete_sequence=["#4a5568"])
fig.update_traces(hovertemplate='%{x}<br>€ %{y:,.2f}')
st.plotly_chart(fig, use_container_width=True)

# Revenue by product family + client
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("CA par famille de produits")
    fig = px.bar(revenue_by_family, x='product_family', y='revenue', color_discrete_sequence=["#4a5568"])
    fig.update_traces(hovertemplate='%{x}<br>€ %{y:,.2f}')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Répartition CA par famille")
    fig = px.pie(revenue_by_family, names='product_family', values='revenue')
    st.plotly_chart(fig, use_container_width=True)

# Top 10 clients
st.markdown("---")
st.subheader("Top 10 clients par CA")
col1, col2 = st.columns(2)

with col1:
    st.dataframe(
        revenue_by_client.rename(columns={'client_name': 'Client', 'revenue': 'CA'}),
        hide_index=True,
        column_config={'CA': st.column_config.NumberColumn(format="€ %,.2f")}
    )

with col2:
    fig = px.bar(revenue_by_client, x='client_name', y='revenue', color_discrete_sequence=["#4a5568"])
    fig.update_traces(hovertemplate='%{x}<br>€ %{y:,.2f}')
    st.plotly_chart(fig, use_container_width=True)

# Client concentration
st.markdown("---")
st.subheader("Concentration du risque client")
revenue_all_clients = filtered_data.groupby('client_name')['revenue'].sum().reset_index()
fig = px.pie(revenue_all_clients, names='client_name', values='revenue')
st.plotly_chart(fig, use_container_width=True)