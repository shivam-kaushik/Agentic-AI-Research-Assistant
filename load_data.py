"""
Load hackathon datasets from GCS bucket into BigQuery.
"""
from google.cloud import bigquery
from google.cloud.exceptions import Conflict
import time

PROJECT_ID = "queryquest-1771952465"
DATASET_ID = "coinvestigator"
DATA_BUCKET = "gs://benchspark-data-1771447466-datasets"


def create_dataset(client: bigquery.Client):
    """Create the BigQuery dataset if it doesn't exist."""
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"

    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"
    dataset.description = "Co-Investigator hackathon data"

    try:
        client.create_dataset(dataset)
        print(f"✅ Created dataset: {DATASET_ID}")
    except Conflict:
        print(f"✅ Dataset already exists: {DATASET_ID}")


def load_table(
    client: bigquery.Client,
    table_name: str,
    gcs_uri: str,
    source_format: str = "CSV",
    skip_rows: int = 1,
    field_delimiter: str = ",",
    schema: list = None,
    autodetect: bool = True,
):
    """Load a table from GCS into BigQuery."""
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    job_config = bigquery.LoadJobConfig()

    if source_format == "CSV":
        job_config.source_format = bigquery.SourceFormat.CSV
        job_config.skip_leading_rows = skip_rows
        job_config.field_delimiter = field_delimiter
        job_config.allow_quoted_newlines = True
        job_config.allow_jagged_rows = True
    elif source_format == "JSON":
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
    elif source_format == "PARQUET":
        job_config.source_format = bigquery.SourceFormat.PARQUET

    if schema:
        job_config.schema = schema
        job_config.autodetect = False
    else:
        job_config.autodetect = autodetect

    job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

    print(f"\n📤 Loading {table_name}...")
    print(f"   Source: {gcs_uri}")

    try:
        load_job = client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)

        # Wait for job to complete
        load_job.result()

        # Get table info
        table = client.get_table(table_ref)
        print(f"   ✅ Loaded {table.num_rows:,} rows into {table_name}")
        return True

    except Exception as e:
        print(f"   ❌ Failed to load {table_name}: {e}")
        return False


def main():
    print("\n🔬 Loading Hackathon Data into BigQuery\n")
    print("=" * 60)

    client = bigquery.Client(project=PROJECT_ID)

    # Step 1: Create dataset
    print("\n1. Creating dataset...")
    create_dataset(client)

    # Step 2: Load tables
    print("\n2. Loading tables from GCS bucket...")

    results = []

    # ClinGen - Gene Disease Validity
    results.append(load_table(
        client,
        table_name="clingen_gene_disease",
        gcs_uri=f"{DATA_BUCKET}/clingen/gene-disease-validity.csv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter=",",
    ))

    # ClinGen - Variant Pathogenicity
    results.append(load_table(
        client,
        table_name="clingen_variant_pathogenicity",
        gcs_uri=f"{DATA_BUCKET}/clingen/variant-pathogenicity.csv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter=",",
    ))

    # ClinGen - Dosage Sensitivity
    results.append(load_table(
        client,
        table_name="clingen_dosage_sensitivity",
        gcs_uri=f"{DATA_BUCKET}/clingen/dosage-sensitivity-genes.csv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter=",",
    ))

    # CIViC - Clinical Evidence
    results.append(load_table(
        client,
        table_name="civic_evidence",
        gcs_uri=f"{DATA_BUCKET}/civic/nightly-AcceptedClinicalEvidenceSummaries.tsv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter="\t",
    ))

    # CIViC - Variants
    results.append(load_table(
        client,
        table_name="civic_variants",
        gcs_uri=f"{DATA_BUCKET}/civic/nightly-VariantSummaries.tsv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter="\t",
    ))

    # CIViC - Assertions
    results.append(load_table(
        client,
        table_name="civic_assertions",
        gcs_uri=f"{DATA_BUCKET}/civic/nightly-AcceptedAssertionSummaries.tsv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter="\t",
    ))

    # Reactome - Pathways
    results.append(load_table(
        client,
        table_name="reactome_pathways",
        gcs_uri=f"{DATA_BUCKET}/reactome/ReactomePathways.txt",
        source_format="CSV",
        skip_rows=0,
        field_delimiter="\t",
        schema=[
            bigquery.SchemaField("pathway_id", "STRING"),
            bigquery.SchemaField("pathway_name", "STRING"),
            bigquery.SchemaField("species", "STRING"),
        ],
        autodetect=False,
    ))

    # Reactome - Pathway Relations
    results.append(load_table(
        client,
        table_name="reactome_pathway_relations",
        gcs_uri=f"{DATA_BUCKET}/reactome/ReactomePathwaysRelation.txt",
        source_format="CSV",
        skip_rows=0,
        field_delimiter="\t",
        schema=[
            bigquery.SchemaField("parent_pathway", "STRING"),
            bigquery.SchemaField("child_pathway", "STRING"),
        ],
        autodetect=False,
    ))

    # Reactome - UniProt Mapping
    results.append(load_table(
        client,
        table_name="reactome_uniprot_mapping",
        gcs_uri=f"{DATA_BUCKET}/reactome/UniProt2Reactome_All_Levels.txt",
        source_format="CSV",
        skip_rows=0,
        field_delimiter="\t",
        schema=[
            bigquery.SchemaField("uniprot_id", "STRING"),
            bigquery.SchemaField("pathway_id", "STRING"),
            bigquery.SchemaField("url", "STRING"),
            bigquery.SchemaField("pathway_name", "STRING"),
            bigquery.SchemaField("evidence_code", "STRING"),
            bigquery.SchemaField("species", "STRING"),
        ],
        autodetect=False,
    ))

    # STRING - Protein Links (large file - might take a while)
    results.append(load_table(
        client,
        table_name="string_protein_links",
        gcs_uri=f"{DATA_BUCKET}/string/9606.protein.links.v12.0.txt.gz",
        source_format="CSV",
        skip_rows=1,
        field_delimiter=" ",
        schema=[
            bigquery.SchemaField("protein1", "STRING"),
            bigquery.SchemaField("protein2", "STRING"),
            bigquery.SchemaField("combined_score", "INTEGER"),
        ],
        autodetect=False,
    ))

    # STRING - Protein Info
    results.append(load_table(
        client,
        table_name="string_protein_info",
        gcs_uri=f"{DATA_BUCKET}/string/9606.protein.info.v12.0.txt.gz",
        source_format="CSV",
        skip_rows=1,
        field_delimiter="\t",
    ))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    success_count = sum(results)
    total_count = len(results)

    print(f"\n  Loaded: {success_count}/{total_count} tables")

    if success_count == total_count:
        print("\n🎉 All data loaded successfully!")
    else:
        print(f"\n⚠️ {total_count - success_count} tables failed to load")

    # List all tables
    print("\n📊 Tables in dataset:")
    tables = list(client.list_tables(f"{PROJECT_ID}.{DATASET_ID}"))
    for table in tables:
        full_table = client.get_table(f"{PROJECT_ID}.{DATASET_ID}.{table.table_id}")
        print(f"   - {table.table_id}: {full_table.num_rows:,} rows")

    print("\n✅ Ready to run the Co-Investigator app!")
    print("   Run: streamlit run app/main.py")


if __name__ == "__main__":
    main()
