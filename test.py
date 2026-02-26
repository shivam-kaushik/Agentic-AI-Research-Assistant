"""
Quick test script to verify BigQuery and GCP access.
"""
from google.cloud import bigquery
from google.cloud import storage

PROJECT_ID = "queryquest-1771952465"
DATA_BUCKET = "benchspark-data-1771447466-datasets"


def test_bigquery():
    """Test BigQuery connection and list datasets."""
    print("=" * 50)
    print("1. Testing BigQuery Connection")
    print("=" * 50)

    try:
        client = bigquery.Client(project=PROJECT_ID)

        # List all datasets
        print(f"\nDatasets in project '{PROJECT_ID}':")
        datasets = list(client.list_datasets())

        if datasets:
            for dataset in datasets:
                print(f"  - {dataset.dataset_id}")

                # List tables in each dataset
                tables = list(client.list_tables(dataset.dataset_id))
                for table in tables[:5]:  # Limit to first 5 tables
                    print(f"      └── {table.table_id}")
                if len(tables) > 5:
                    print(f"      └── ... and {len(tables) - 5} more tables")
        else:
            print("  No datasets found (you may need to create one)")

        # Test a simple query
        print("\nTesting query execution...")
        query = "SELECT 1 as test_value, 'BigQuery works!' as message"
        result = client.query(query).result()

        for row in result:
            print(f"  ✅ Query successful: {row.message}")

        return True

    except Exception as e:
        print(f"  ❌ BigQuery error: {e}")
        return False


def test_storage():
    """Test Cloud Storage access to hackathon data bucket."""
    print("\n" + "=" * 50)
    print("2. Testing Cloud Storage Access")
    print("=" * 50)

    try:
        client = storage.Client(project=PROJECT_ID)
        bucket = client.bucket(DATA_BUCKET)

        print(f"\nListing folders in gs://{DATA_BUCKET}/")

        # List top-level folders
        blobs = client.list_blobs(DATA_BUCKET, delimiter='/')

        # Get prefixes (folders)
        folders = []
        for page in blobs.pages:
            folders.extend(page.prefixes)

        if folders:
            for folder in sorted(folders):
                print(f"  📁 {folder}")
            print(f"\n  ✅ Found {len(folders)} data folders")
        else:
            print("  ⚠️ No folders found or no access")

        return True

    except Exception as e:
        print(f"  ❌ Storage error: {e}")
        return False


def test_sample_data():
    """Try to query some actual data if available."""
    print("\n" + "=" * 50)
    print("3. Testing Sample Data Query")
    print("=" * 50)

    client = bigquery.Client(project=PROJECT_ID)

    # Try common dataset names that might exist
    test_queries = [
        ("coinvestigator.clingen_gene_disease", "SELECT * FROM `{table}` LIMIT 3"),
        ("benchspark.clingen", "SELECT * FROM `{table}` LIMIT 3"),
        ("hackathon.clingen", "SELECT * FROM `{table}` LIMIT 3"),
    ]

    for table_name, query_template in test_queries:
        full_table = f"{PROJECT_ID}.{table_name}"
        try:
            query = query_template.format(table=full_table)
            result = client.query(query).result()
            rows = list(result)

            if rows:
                print(f"\n  ✅ Found data in {table_name}!")
                print(f"     Sample columns: {list(rows[0].keys())[:5]}")
                return True

        except Exception as e:
            # Table doesn't exist, try next
            continue

    print("\n  ⚠️ No pre-loaded datasets found.")
    print("     You may need to load data from the GCS bucket.")
    return False


def main():
    print("\n🔬 Co-Investigator GCP Access Test\n")

    bq_ok = test_bigquery()
    storage_ok = test_storage()
    data_ok = test_sample_data()

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  BigQuery Access:  {'✅ OK' if bq_ok else '❌ Failed'}")
    print(f"  Storage Access:   {'✅ OK' if storage_ok else '❌ Failed'}")
    print(f"  Sample Data:      {'✅ Found' if data_ok else '⚠️ Need to load'}")

    if bq_ok and storage_ok:
        print("\n🎉 GCP access is working!")
        if not data_ok:
            print("\nNext step: Load data into BigQuery from the GCS bucket.")
            print("Run: python load_data.py")
    else:
        print("\n⚠️ Some access issues detected. Check errors above.")


if __name__ == "__main__":
    main()
