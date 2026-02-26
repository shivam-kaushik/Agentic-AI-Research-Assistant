"""Debug BigQuery tables to see actual schema and data."""
from google.cloud import bigquery

PROJECT_ID = "queryquest-1771952465"
DATASET_ID = "coinvestigator"

client = bigquery.Client(project=PROJECT_ID)

tables_to_check = [
    "clingen_gene_disease",
    "civic_evidence",
    "reactome_pathways",
]

for table_name in tables_to_check:
    print(f"\n{'='*60}")
    print(f"TABLE: {table_name}")
    print('='*60)

    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    # Get schema
    try:
        table = client.get_table(table_ref)
        print("\nSchema:")
        for field in table.schema[:10]:
            print(f"  - {field.name}: {field.field_type}")
    except Exception as e:
        print(f"Schema error: {e}")

    # Get sample data
    try:
        query = f"SELECT * FROM `{table_ref}` LIMIT 3"
        result = client.query(query).result()
        rows = list(result)

        print(f"\nSample data ({len(rows)} rows):")
        for i, row in enumerate(rows):
            print(f"\n  Row {i+1}:")
            row_dict = dict(row)
            for k, v in list(row_dict.items())[:5]:
                print(f"    {k}: {str(v)[:50]}")
    except Exception as e:
        print(f"Query error: {e}")
