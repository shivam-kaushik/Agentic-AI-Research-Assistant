import os
from google.cloud import bigquery
import pandas as pd
from config.gcp_config import config, tables

client = bigquery.Client(project=config.project_id)

table_names = [
    tables.clingen_gene_disease, 
    tables.civic_evidence, 
    tables.reactome_pathways
]

for t in table_names:
    print(f"\n--- Checking table {t} ---")
    try:
        query = f"SELECT * FROM `{config.project_id}.{t}` LIMIT 1"
        df = client.query(query).result().to_dataframe()
        print("Columns:", list(df.columns))
        print("First row:", df.to_dict(orient="records"))
    except Exception as e:
        print("Error:", e)
