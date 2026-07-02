# Prévisions CA

# Library import
import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Page config
st.set_page_config(page_title='Prévisions CA', layout='wide')

# Data loading
if 'data' not in st.session_state:
    st.session_state['data'] = pd.read_csv("dataset_industry.csv")
data = st.session_state['data']
data['order_date'] = pd.to_datetime(data['order_date'])

# Sidebar filters
st.sidebar.header("Filtres")
selected_families = st.sidebar.multiselect("Famille de produits", options=data['product_family'].unique(), default=data['product_family'].unique())
selected_clients = st.sidebar.multiselect("Client", options=data['client_name'].unique(), default=data['client_name'].unique())
forecast_months = st.sidebar.slider("Nombre de mois à prévoir", min_value=3, max_value=24, value=12)

# Filtered data
filtered_data = data[
    (data['product_family'].isin(selected_families)) &
    (data['client_name'].isin(selected_clients))
]

# Prepare Prophet data
revenue_by_month = filtered_data.groupby(filtered_data['order_date'].dt.to_period('M'))['revenue'].sum().reset_index()
revenue_by_month['order_date'] = revenue_by_month['order_date'].dt.to_timestamp()
prophet_df = revenue_by_month.rename(columns={'order_date': 'ds', 'revenue': 'y'})

# Train Prophet model
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(prophet_df)

# Forecast
future = model.make_future_dataframe(periods=forecast_months, freq='MS')
forecast = model.predict(future)

# Forecast summary
forecast_period = forecast[forecast['ds'] > prophet_df['ds'].max()]
total_forecast = forecast_period['yhat'].sum()
avg_monthly_forecast = forecast_period['yhat'].mean()
last_year_revenue = prophet_df.tail(12)['y'].sum()
growth_pct = ((total_forecast / last_year_revenue) - 1) * 100 if forecast_months == 12 else None

# System prompt
system_prompt = f"""Tu es un analyste financier s'adressant à la direction d'une TPE/PME. Ton ton est professionnel et orienté action.

Voici les données de prévision CA :
- CA total prévu sur les {forecast_months} prochains mois : {total_forecast:,.2f} €
- CA mensuel moyen prévu : {avg_monthly_forecast:,.2f} €
- CA des 12 derniers mois (historique) : {last_year_revenue:,.2f} €
{f"- Croissance estimée vs année précédente : {growth_pct:.1f}%" if growth_pct else ""}
- Détail des prévisions : {forecast_period[['ds','yhat','yhat_lower','yhat_upper']].to_string()}

Ton analyse doit :
- Commenter la tendance prévue du CA
- Identifier les mois ou périodes à fort/faible potentiel
- Alerter si la tendance est préoccupante
- Proposer des pistes pour maximiser le CA sur les périodes creuses

Réponds de manière concise, structurée et avec des recommandations actionnables."""

# AI Chat
st.sidebar.markdown("---")
st.sidebar.subheader("Analyse IA")

if 'messages_forecast' not in st.session_state:
    st.session_state['messages_forecast'] = []

user_question = st.sidebar.text_input("Posez une question sur les prévisions...")

if user_question:
    st.session_state['messages_forecast'].append({"role": "user", "content": user_question})
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system_prompt,
        messages=st.session_state['messages_forecast']
    )
    st.session_state['messages_forecast'].append({"role": "assistant", "content": response.content[0].text})

for msg in st.session_state['messages_forecast']:
    if msg['role'] == 'user':
        st.sidebar.write("**Vous :** " + msg['content'])
    else:
        st.sidebar.write("**IA :** " + msg['content'])

# Page title
st.title('Prévisions CA')

# Global forecast KPIs
col1, col2, col3 = st.columns(3)
col1.metric("CA prévu", f"{total_forecast:,.2f} €")
col2.metric("Moyenne mensuelle prévue", f"{avg_monthly_forecast:,.2f} €")
if growth_pct:
    col3.metric("Croissance vs année précédente", f"{growth_pct:.1f}%")

# Forecast chart
st.markdown("---")
st.subheader("Prévisions vs Historique")
fig = px.line()
fig.add_scatter(x=prophet_df['ds'], y=prophet_df['y'], name='Historique', line=dict(color='#4a5568'))
fig.add_scatter(x=forecast['ds'], y=forecast['yhat'], name='Prévision', line=dict(color='#a0aec0', dash='dash'))
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Borne haute', line=dict(color='lightgrey', dash='dot'))
fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Borne basse', line=dict(color='lightgrey', dash='dot'))
fig.update_traces(hovertemplate='%{x}<br>€ %{y:,.2f}')
st.plotly_chart(fig, use_container_width=True)

# Forecast table
st.markdown("---")
st.subheader("Tableau des prévisions mensuelles")
forecast_table = forecast_period[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)
forecast_table.columns = ['Mois', 'Prévision', 'Borne basse', 'Borne haute']
forecast_table['Mois'] = forecast_table['Mois'].dt.strftime('%B %Y')
st.dataframe(
    forecast_table,
    hide_index=True,
    column_config={
        'Prévision': st.column_config.NumberColumn(format="€ %,.2f"),
        'Borne basse': st.column_config.NumberColumn(format="€ %,.2f"),
        'Borne haute': st.column_config.NumberColumn(format="€ %,.2f"),
    }
)