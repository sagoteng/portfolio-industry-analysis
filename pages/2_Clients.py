# Analyse clients

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
st.set_page_config(page_title='Analyse Clients', layout='wide')

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
revenue_by_client = filtered_data.groupby('client_name')['revenue'].sum().reset_index().sort_values('revenue', ascending=False)
orders_by_client = filtered_data.groupby('client_name')['revenue'].count().reset_index().rename(columns={'revenue': 'nb_orders'})
last_order_by_client = filtered_data.groupby('client_name')['order_date'].max().reset_index().rename(columns={'order_date': 'last_order'})

client_summary = revenue_by_client.merge(orders_by_client, on='client_name').merge(last_order_by_client, on='client_name')
client_summary['avg_order_value'] = (client_summary['revenue'] / client_summary['nb_orders']).round(2)
client_summary['days_since_last_order'] = (pd.Timestamp.now() - client_summary['last_order']).dt.days

# Global client KPIs
total_clients = client_summary['client_name'].nunique()
avg_revenue_per_client = client_summary['revenue'].mean()
top_client = client_summary.iloc[0]['client_name']
top_client_share = (client_summary.iloc[0]['revenue'] / client_summary['revenue'].sum() * 100)

# AI system prompt
system_prompt = f"""Tu es un analyste commercial s'adressant à la direction d'une TPE/PME. Ton ton est professionnel et orienté action.

Voici les données clients :
- Nombre de clients : {total_clients}
- CA moyen par client : {avg_revenue_per_client:,.2f} €
- Top client : {top_client} ({top_client_share:.1f}% du CA)
- Résumé clients : {client_summary[['client_name','revenue','nb_orders','days_since_last_order']].to_string()}

Ton analyse doit :
- Identifier les clients en perte de vitesse (baisse de commandes, longue inactivité)
- Alerter sur la concentration du risque (dépendance à un ou deux clients majeurs)
- Proposer des actions de fidélisation pour les meilleurs clients
- Suggérer des pistes de développement commercial sur les clients à fort potentiel

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
st.title('Analyse Clients')

# Global client KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nombre de clients", f"{total_clients}")
col2.metric("CA moyen par client", f"{avg_revenue_per_client:,.2f} €")
col3.metric("Top client", top_client)
col4.metric("Part du top client", f"{top_client_share:.1f}%")

# Revenue by client bar chart
st.markdown("---")
st.subheader("CA par client")
fig = px.bar(revenue_by_client, x='client_name', y='revenue', color_discrete_sequence=["#4a5568"])
fig.update_traces(hovertemplate='%{x}<br>€ %{y:,.2f}')
fig.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig, use_container_width=True)

# Client summary table
st.markdown("---")
st.subheader("Tableau récapitulatif clients")
st.dataframe(
    client_summary.rename(columns={
        'client_name': 'Client',
        'revenue': 'CA Total',
        'nb_orders': 'Nb Commandes',
        'last_order': 'Dernière Commande',
        'avg_order_value': 'Valeur Moy. Commande',
        'days_since_last_order': 'Jours depuis dernière commande'
    }),
    hide_index=True,
    column_config={
        'CA Total': st.column_config.NumberColumn(format="€ %,.2f"),
        'Valeur Moy. Commande': st.column_config.NumberColumn(format="€ %,.2f"),
    }
)

# Fidélité
st.markdown("---")
st.subheader("Fidélité client — Nombre de commandes")
fig = px.bar(orders_by_client.sort_values('nb_orders', ascending=False),
             x='client_name', y='nb_orders', color_discrete_sequence=["#4a5568"])
fig.update_layout(xaxis_tickangle=-45)
fig.update_traces(hovertemplate='%{x}<br>%{y} commandes')
st.plotly_chart(fig, use_container_width=True)

# Recency
st.markdown("---")
st.subheader("Récence — Jours depuis la dernière commande")
fig = px.bar(client_summary.sort_values('days_since_last_order', ascending=True),
             x='client_name', y='days_since_last_order', color_discrete_sequence=["#4a5568"])
fig.update_layout(xaxis_tickangle=-45)
fig.update_traces(hovertemplate='%{x}<br>%{y} jours')
st.plotly_chart(fig, use_container_width=True)