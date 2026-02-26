"""
GCP Setup Script for Co-Investigator
- Creates BigQuery external tables (queries GCS directly, no data copying)
- Creates Firestore collections
- Tests all connections
"""
from google.cloud import bigquery
from google.cloud import firestore
from google.cloud.exceptions import Conflict, NotFound
import warnings

# Suppress the quota warning
warnings.filterwarnings("ignore", message=".*quota.*")

PROJECT_ID = "queryquest-1771952465"
DATASET_ID = "coinvestigator"
DATA_BUCKET = "gs://benchspark-data-1771447466-datasets"


def create_dataset(client: bigquery.Client):
    """Create BigQuery dataset if it doesn't exist."""
    dataset_ref = f"{PROJECT_ID}.{DATASET_ID}"
    dataset = bigquery.Dataset(dataset_ref)
    dataset.location = "US"

    try:
        client.create_dataset(dataset)
        print(f"✅ Created dataset: {DATASET_ID}")
    except Conflict:
        print(f"✅ Dataset already exists: {DATASET_ID}")


def create_external_table(
    client: bigquery.Client,
    table_name: str,
    gcs_uri: str,
    source_format: str = "CSV",
    skip_rows: int = 1,
    field_delimiter: str = ",",
    schema: list = None,
):
    """
    Create an external table that queries GCS directly.
    No data is copied - queries run against the source files.
    """
    table_ref = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"

    # Define external config
    external_config = bigquery.ExternalConfig(source_format)
    external_config.source_uris = [gcs_uri]
    external_config.autodetect = True if schema is None else False

    if source_format == "CSV":
        external_config.options.skip_leading_rows = skip_rows
        external_config.options.field_delimiter = field_delimiter
        external_config.options.allow_quoted_newlines = True

    # Create table with external config
    table = bigquery.Table(table_ref)
    table.external_data_configuration = external_config

    if schema:
        table.schema = schema

    try:
        # Delete if exists, then create
        try:
            client.delete_table(table_ref)
        except NotFound:
            pass

        client.create_table(table)
        print(f"✅ Created external table: {table_name} → {gcs_uri}")
        return True

    except Exception as e:
        print(f"❌ Failed to create {table_name}: {e}")
        return False


def setup_bigquery():
    """Set up all BigQuery external tables."""
    print("\n" + "=" * 60)
    print("1. Setting up BigQuery External Tables")
    print("=" * 60)
    print("(These query GCS directly - no data copying!)\n")

    client = bigquery.Client(project=PROJECT_ID)

    # Create dataset
    create_dataset(client)

    results = []

    # ClinGen - Gene Disease Validity
    results.append(create_external_table(
        client,
        table_name="clingen_gene_disease",
        gcs_uri=f"{DATA_BUCKET}/clingen/gene-disease-validity.csv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter=",",
    ))

    # ClinGen - Variant Pathogenicity
    results.append(create_external_table(
        client,
        table_name="clingen_variant_pathogenicity",
        gcs_uri=f"{DATA_BUCKET}/clingen/variant-pathogenicity.csv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter=",",
    ))

    # ClinGen - Dosage Sensitivity
    results.append(create_external_table(
        client,
        table_name="clingen_dosage_sensitivity",
        gcs_uri=f"{DATA_BUCKET}/clingen/dosage-sensitivity-genes.csv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter=",",
    ))

    # CIViC - Clinical Evidence
    results.append(create_external_table(
        client,
        table_name="civic_evidence",
        gcs_uri=f"{DATA_BUCKET}/civic/nightly-AcceptedClinicalEvidenceSummaries.tsv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter="\t",
    ))

    # CIViC - Variants
    results.append(create_external_table(
        client,
        table_name="civic_variants",
        gcs_uri=f"{DATA_BUCKET}/civic/nightly-VariantSummaries.tsv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter="\t",
    ))

    # CIViC - Assertions
    results.append(create_external_table(
        client,
        table_name="civic_assertions",
        gcs_uri=f"{DATA_BUCKET}/civic/nightly-AcceptedAssertionSummaries.tsv",
        source_format="CSV",
        skip_rows=1,
        field_delimiter="\t",
    ))

    # Reactome - Pathways
    results.append(create_external_table(
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
    ))

    # Reactome - Pathway Relations
    results.append(create_external_table(
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
    ))

    # Reactome - UniProt Mapping
    results.append(create_external_table(
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
    ))

    # STRING - Protein Links (using .gz file directly)
    results.append(create_external_table(
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
    ))

    # STRING - Protein Info
    results.append(create_external_table(
        client,
        table_name="string_protein_info",
        gcs_uri=f"{DATA_BUCKET}/string/9606.protein.info.v12.0.txt.gz",
        source_format="CSV",
        skip_rows=1,
        field_delimiter="\t",
    ))

    print(f"\n📊 Created {sum(results)}/{len(results)} external tables")
    return all(results)


def setup_firestore():
    """Set up Firestore collections."""
    print("\n" + "=" * 60)
    print("2. Setting up Firestore")
    print("=" * 60)

    try:
        db = firestore.Client(project=PROJECT_ID)

        # Create initial documents to establish collections
        # (Firestore creates collections automatically when you add documents)

        # Test write to agent_sessions
        test_doc = db.collection("agent_sessions").document("_init")
        test_doc.set({
            "initialized": True,
            "description": "Co-Investigator session storage"
        })
        print("✅ Created collection: agent_sessions")

        # Test write to hitl_checkpoints
        test_doc = db.collection("hitl_checkpoints").document("_init")
        test_doc.set({
            "initialized": True,
            "description": "HITL checkpoint storage"
        })
        print("✅ Created collection: hitl_checkpoints")

        return True

    except Exception as e:
        print(f"❌ Firestore setup failed: {e}")
        print("\n💡 You may need to create the Firestore database manually:")
        print("   https://console.cloud.google.com/firestore?project=queryquest-1771952465")
        return False


def test_bigquery_query():
    """Test querying the external tables."""
    print("\n" + "=" * 60)
    print("3. Testing BigQuery Queries")
    print("=" * 60)

    client = bigquery.Client(project=PROJECT_ID)

    test_queries = [
        ("clingen_gene_disease", "SELECT * FROM `{table}` LIMIT 5"),
        ("civic_evidence", "SELECT * FROM `{table}` LIMIT 5"),
        ("reactome_pathways", "SELECT * FROM `{table}` WHERE species = 'Homo sapiens' LIMIT 5"),
    ]

    for table_name, query_template in test_queries:
        full_table = f"{PROJECT_ID}.{DATASET_ID}.{table_name}"
        query = query_template.format(table=full_table)

        try:
            result = client.query(query).result()
            rows = list(result)
            print(f"✅ {table_name}: {len(rows)} rows returned")

            # Show sample columns
            if rows:
                cols = list(rows[0].keys())[:4]
                print(f"   Columns: {', '.join(cols)}...")

        except Exception as e:
            print(f"❌ {table_name}: {e}")


def test_vertex_ai():
    """Test Vertex AI connection."""
    print("\n" + "=" * 60)
    print("4. Testing Vertex AI")
    print("=" * 60)

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        vertexai.init(project=PROJECT_ID, location="us-central1")
        model = GenerativeModel("gemini-2.5-pro")

        response = model.generate_content("Say 'GCP setup complete!' in exactly those words.")
        print(f"✅ Vertex AI connected")
        print(f"   Model response: {response.text.strip()}")
        return True

    except Exception as e:
        print(f"❌ Vertex AI error: {e}")
        return False


def main():
    print("\n🔬 Co-Investigator GCP Setup\n")

    # Step 1: BigQuery external tables
    bq_ok = setup_bigquery()

    # Step 2: Firestore
    fs_ok = setup_firestore()

    # Step 3: Test queries
    test_bigquery_query()

    # Step 4: Test Vertex AI
    vai_ok = test_vertex_ai()

    # Summary
    print("\n" + "=" * 60)
    print("SETUP COMPLETE")
    print("=" * 60)
    print(f"  BigQuery External Tables: {'✅ OK' if bq_ok else '❌ Failed'}")
    print(f"  Firestore Collections:    {'✅ OK' if fs_ok else '❌ Failed'}")
    print(f"  Vertex AI:                {'✅ OK' if vai_ok else '❌ Failed'}")

    if bq_ok and fs_ok and vai_ok:
        print("\n🎉 All GCP services configured!")
        print("\nRun the app:")
        print("  streamlit run app/main.py")
    else:
        print("\n⚠️ Some services need attention. Check errors above.")


if __name__ == "__main__":
    main()
