import json
from google.cloud import bigquery
from config.gcp_config import config, tables

client = bigquery.Client(project=config.project_id)

queries = {
    "clingen": f"SELECT COUNT(*) as c FROM `{config.project_id}.{tables.clingen_gene_disease}` WHERE LOWER(string_field_2) LIKE '%ipf%' OR LOWER(string_field_2) LIKE '%fibrosis%' OR LOWER(string_field_2) LIKE '%idiopathic%'",
    "civic": f"SELECT COUNT(*) as c FROM `{config.project_id}.{tables.civic_evidence}` WHERE LOWER(disease) LIKE '%ipf%' OR LOWER(disease) LIKE '%fibrosis%' OR LOWER(disease) LIKE '%idiopathic%'",
    "reactome": f"SELECT COUNT(*) as c FROM `{config.project_id}.{tables.reactome_pathways}` WHERE LOWER(pathway_name) LIKE '%ipf%' OR LOWER(pathway_name) LIKE '%fibrosis%' OR LOWER(pathway_name) LIKE '%idiopathic%'"
}

results = {}
for k, v in queries.items():
    res = list(client.query(v).result())
    results[k] = dict(res[0])

with open("ipf_counts.json", "w") as f:
    json.dump(results, f)
