import os
import json
from google.cloud import bigquery
import pandas as pd
from config.gcp_config import config, tables

client = bigquery.Client(project=config.project_id)

table_names = [
    tables.clingen_gene_disease, 
    tables.clingen_variant_pathogenicity,
    tables.civic_evidence, 
    tables.reactome_pathways
]

out = {}
for t in table_names:
    try:
        query = f"SELECT * FROM `{config.project_id}.{t}` LIMIT 1"
        df = client.query(query).result().to_dataframe()
        out[t] = {
            "columns": list(df.columns),
            "row": df.to_dict(orient="records")[0] if len(df) > 0 else None
        }
    except Exception as e:
        out[t] = {"error": str(e)}

with open("bq_schemas.json", "w") as f:
    json.dump(out, f, indent=2)
