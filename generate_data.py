import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Parameters
n = 2000
start_date = datetime(2022, 1, 1)
end_date = datetime(2024, 12, 31)

# Client pool (B2B - 30 unique clients)
client_pool = [
    ("CL001", "Renault"),
    ("CL002", "Stellantis"),
    ("CL003", "Airbus"),
    ("CL004", "Safran"),
    ("CL005", "Thales"),
    ("CL006", "Schneider Electric"),
    ("CL007", "Faurecia"),
    ("CL008", "Valeo"),
    ("CL009", "Saint-Gobain"),
    ("CL010", "Legrand"),
    ("CL011", "Total Energies"),
    ("CL012", "Alstom"),
    ("CL013", "Michelin"),
    ("CL014", "Plastic Omnium"),
    ("CL015", "Forvia"),
    ("CL016", "Soitec"),
    ("CL017", "Dassault Aviation"),
    ("CL018", "Naval Group"),
    ("CL019", "MBDA"),
    ("CL020", "Hutchinson"),
    ("CL021", "Lisi Group"),
    ("CL022", "Mecachrome"),
    ("CL023", "Figeac Aero"),
    ("CL024", "Aubert & Duval"),
    ("CL025", "Manoir Industries"),
    ("CL026", "Hydro Extrusion"),
    ("CL027", "Constellium"),
    ("CL028", "Aperam"),
    ("CL029", "Timet"),
    ("CL030", "Ugitech"),
]

product_families = ['Découpage', 'Emboutissage', 'Tôlerie', 'Chaudronnerie']

# Unit price ranges by product family
price_ranges = {
    'Découpage': (0.5, 5.0),
    'Emboutissage': (1.0, 10.0),
    'Tôlerie': (50.0, 500.0),
    'Chaudronnerie': (200.0, 2000.0),
}

cost_ranges = {
    'Découpage': (0.3, 3.5),
    'Emboutissage': (0.7, 7.0),
    'Tôlerie': (35.0, 350.0),
    'Chaudronnerie': (140.0, 1400.0),
}

# Generate data
np.random.seed(42)

order_dates = [start_date + timedelta(days=int(np.random.randint(0, 1095))) for _ in range(n)]
delivery_dates = [od + timedelta(days=int(np.random.randint(7, 60))) for od in order_dates]

client_choices = [client_pool[i] for i in np.random.randint(0, len(client_pool), n)]
client_ids = [c[0] for c in client_choices]
client_names = [c[1] for c in client_choices]

product_families_col = np.random.choice(product_families, n)

quantities = np.random.randint(10, 5000, n)

unit_prices = np.array([
    round(np.random.uniform(*price_ranges[pf]), 2)
    for pf in product_families_col
])

unit_costs = np.array([
    round(np.random.uniform(*cost_ranges[pf]), 2)
    for pf in product_families_col
])

revenues = np.round(quantities * unit_prices, 2)

margins = np.round(revenues - (quantities * unit_costs), 2)
margin_rates = np.round((margins / revenues) * 100, 2)

on_time = np.random.choice(['Yes', 'No'], n, p=[0.85, 0.15])
non_conformity = np.random.choice(['Yes', 'No'], n, p=[0.05, 0.95])

data = {
    'order_date': order_dates,
    'delivery_date': delivery_dates,
    'client_id': client_ids,
    'client_name': client_names,
    'product_family': product_families_col,
    'quantity': quantities,
    'unit_price': unit_prices,
    'revenue': revenues,
    'on_time': on_time,
    'non_conformity': non_conformity,
    'unit_cost': unit_costs,
    'margin': margins,
    'margin_rate': margin_rates,
}

df = pd.DataFrame(data)
df.to_csv("dataset_industry.csv", index=False)
print(f"Dataset generated: {n} orders exported.")