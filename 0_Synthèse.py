# Page Synthèse

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
st.set_page_config(page_title='Synthèse', layout='wide')

# File uploader
uploaded_file = st.sidebar.file_uploader("Charger vos données (CSV ou Excel)", type=["csv", "xlsx"])
if uploaded_file is not None:
    if uploaded_file.name.endswith('.xlsx'):
        st.session_state['data'] = pd.read_excel(uploaded_file)
    else:
        st.session_state['data'] = pd.read_csv(uploaded_file)
else:
    if 'data' not in st.session_state:
        st.session_state['data'] = pd.read_csv("dataset_industry.csv")
data = st.session_state['data']
data['order_date'] = pd.to_datetime(data['order_date'])

# Data loading
if 'data' not in st.session_state:
    st.session_state['data'] = pd.read_csv("dataset_industry.csv")
data = st.session_state['data']
data['order_date'] = pd.to_datetime(data['order_date'])

# Sidebar filters
st.sidebar.markdown("---")
st.sidebar.header("Filtres")
last_year = data['order_date'].dt.year.max()
default_start = pd.Timestamp(f"{last_year}-01-01")
default_end = data['order_date'].max()

start = st.sidebar.date_input("Date de début", value=default_start)
end = st.sidebar.date_input("Date de fin", value=default_end)
selected_families = st.sidebar.multiselect("Famille de produits", options=data['product_family'].unique(), default=data['product_family'].unique())
selected_clients = st.sidebar.multiselect("Client", options=data['client_name'].unique(), default=data['client_name'].unique())

# Filtered data
filtered_data = data[
    (data['order_date'] >= pd.Timestamp(start)) &
    (data['order_date'] <= pd.Timestamp(end)) &
    (data['product_family'].isin(selected_families)) &
    (data['client_name'].isin(selected_clients))
]

# Période précédente
period_duration = (pd.Timestamp(end) - pd.Timestamp(start)).days
prev_start = pd.Timestamp(start) - pd.Timedelta(days=period_duration)
prev_end = pd.Timestamp(start) - pd.Timedelta(days=1)
prev_data = data[
    (data['order_date'] >= prev_start) &
    (data['order_date'] <= prev_end) &
    (data['product_family'].isin(selected_families)) &
    (data['client_name'].isin(selected_clients))
]

# KPIs
total_revenue = filtered_data['revenue'].sum()
total_orders = len(filtered_data)
otd_rate = (filtered_data['on_time'] == 'Yes').mean() * 100
non_conformity_rate = (filtered_data['non_conformity'] == 'Yes').mean() * 100
avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
total_clients = filtered_data['client_name'].nunique()
top_client = filtered_data.groupby('client_name')['revenue'].sum().idxmax()
top_client_share = (filtered_data.groupby('client_name')['revenue'].sum().max() / total_revenue * 100)
best_family = filtered_data.groupby('product_family')['revenue'].sum().idxmax()
total_margin = filtered_data['margin'].sum()
avg_margin_rate = filtered_data['margin_rate'].mean()
prev_margin = prev_data['margin'].sum() if len(prev_data) > 0 else 0
delta_margin = total_margin - prev_margin

# Deltas
prev_revenue = prev_data['revenue'].sum()
prev_orders = len(prev_data)
prev_otd = (prev_data['on_time'] == 'Yes').mean() * 100 if len(prev_data) > 0 else 0
prev_nc = (prev_data['non_conformity'] == 'Yes').mean() * 100 if len(prev_data) > 0 else 0
prev_avg_order = prev_data['revenue'].sum() / len(prev_data) if len(prev_data) > 0 else 0

delta_revenue = total_revenue - prev_revenue
delta_orders = total_orders - prev_orders
delta_otd = otd_rate - prev_otd
delta_nc = non_conformity_rate - prev_nc
delta_avg_order = avg_order_value - prev_avg_order
prev_avg_margin_rate = prev_data['margin_rate'].mean() if len(prev_data) > 0 else 0
prev_clients = prev_data['client_name'].nunique() if len(prev_data) > 0 else 0

delta_margin_rate = avg_margin_rate - prev_avg_margin_rate
delta_clients = total_clients - prev_clients

# CA par mois
revenue_by_month = filtered_data.groupby(filtered_data['order_date'].dt.to_period('M'))['revenue'].sum().reset_index()
revenue_by_month['order_date'] = revenue_by_month['order_date'].astype(str)

# CA par famille
revenue_by_family = filtered_data.groupby('product_family')['revenue'].sum().reset_index()

# Top 5 clients
top5_clients = filtered_data.groupby('client_name')['revenue'].sum().reset_index().sort_values('revenue', ascending=False).head(5)

# System prompt
system_prompt = f"""Tu es un analyste de données industrielles s'adressant à la direction d'une TPE/PME. Ton ton est professionnel et orienté action.

Voici la synthèse des indicateurs clés :
- CA total : {total_revenue:,.2f} € (variation : {delta_revenue:,.2f} €)
- Nombre de commandes : {total_orders} (variation : {delta_orders:+.0f})
- Taux OTD : {otd_rate:.1f}% (variation : {delta_otd:.1f}%)
- Taux non-conformité : {non_conformity_rate:.1f}% (variation : {delta_nc:.1f}%)
- Valeur moyenne commande : {avg_order_value:,.2f} €
- Nombre de clients actifs : {total_clients}
- Top client : {top_client} ({top_client_share:.1f}% du CA)
- Famille la plus performante : {best_family}

Fais une synthèse exécutive concise (5-8 lignes max) avec les points clés et 2-3 recommandations prioritaires."""

# AI Chat
st.sidebar.markdown("---")
st.sidebar.subheader("Analyse IA")

if 'messages_synthese' not in st.session_state:
    st.session_state['messages_synthese'] = []

user_question = st.sidebar.text_input("Posez une question...")
chat_container = st.sidebar.container()

if user_question:
    st.session_state['messages_synthese'].append({"role": "user", "content": user_question})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system_prompt,
        messages=st.session_state['messages_synthese']
    )
    st.session_state['messages_synthese'].append({"role": "assistant", "content": response.content[0].text})

for msg in st.session_state['messages_synthese']:
    if msg['role'] == 'user':
        st.sidebar.write("**Vous :** " + msg['content'])
    else:
        st.sidebar.write("**IA :** " + msg['content'])

# Page title
st.title('Synthèse Exécutive')
st.caption(f"Période : {start} → {end}")

# KPIs row 1
st.markdown("---")
st.subheader("Indicateurs clés")
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Chiffre d'affaires", f"{total_revenue:,.2f} €", delta=f"{delta_revenue:,.2f} €")
col2.metric("Nb commandes", f"{total_orders:,}", delta=f"{delta_orders:+.0f}")
col3.metric("Valeur moy. commande", f"{avg_order_value:,.2f} €", delta=f"{delta_avg_order:,.2f} €")
col4.metric("Taux OTD", f"{otd_rate:.1f}%", delta=f"{delta_otd:.1f}%")
col5.metric("Taux non-conformité", f"{non_conformity_rate:.1f}%", delta=f"{delta_nc:.1f}%", delta_color="inverse")

#KPIs row 2
st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.metric("Marge totale", f"{total_margin:,.2f} €", delta=f"{delta_margin:,.2f} €")
col2.metric("Taux de marge moyen", f"{avg_margin_rate:.1f}%", delta=f"{delta_margin_rate:.1f}%")
col3.metric("Nb clients actifs", f"{total_clients}", delta=f"{delta_clients:+.0f}")

# Charts
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("CA par mois")
    fig = px.line(revenue_by_month, x='order_date', y='revenue', color_discrete_sequence=["#4a5568"])
    fig.update_traces(hovertemplate='%{x}<br>€ %{y:,.2f}')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Répartition CA par famille")
    fig = px.pie(revenue_by_family, names='product_family', values='revenue')
    st.plotly_chart(fig, use_container_width=True)

# Top 5 clients + alertes
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 5 clients")
    st.dataframe(
        top5_clients.rename(columns={'client_name': 'Client', 'revenue': 'CA'}),
        hide_index=True,
        column_config={'CA': st.column_config.NumberColumn(format="€ %,.2f")}
    )

with col2:
    st.subheader("Alertes")
    if otd_rate < 95:
        st.error(f"⚠️ Taux OTD sous l'objectif de 95% : {otd_rate:.1f}%")
    else:
        st.success(f"✅ Taux OTD dans les objectifs : {otd_rate:.1f}%")
    
    if non_conformity_rate > 5:
        st.error(f"⚠️ Taux de non-conformité élevé : {non_conformity_rate:.1f}%")
    else:
        st.success(f"✅ Taux de non-conformité maîtrisé : {non_conformity_rate:.1f}%")
    
    if top_client_share > 30:
        st.warning(f"⚠️ Concentration client élevée : {top_client} représente {top_client_share:.1f}% du CA")
    else:
        st.success(f"✅ Risque de concentration client maîtrisé")
    
    if delta_revenue < 0:
        st.error(f"⚠️ Baisse du CA vs période précédente : {delta_revenue:,.2f} €")
    else:
        st.success(f"✅ CA en hausse vs période précédente : +{delta_revenue:,.2f} €")
    if avg_margin_rate < 15:
        st.error(f"⚠️ Taux de marge insuffisant : {avg_margin_rate:.1f}% (objectif marché : 15-25%)")
    elif avg_margin_rate < 20:
        st.warning(f"⚠️ Taux de marge en dessous de la moyenne du secteur : {avg_margin_rate:.1f}% (objectif : 20-25%)")
    else:
        st.success(f"✅ Taux de marge dans les objectifs du secteur : {avg_margin_rate:.1f}%")