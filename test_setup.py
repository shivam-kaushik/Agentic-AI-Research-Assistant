"""
Test script to verify the setup is working correctly.
Run this after completing GCP setup and before running the main app.
"""
import os
import sys

def check_env():
    """Check environment variables."""
    print("=" * 50)
    print("1. Checking Environment Variables")
    print("=" * 50)

    required = ["GOOGLE_CLOUD_PROJECT", "GOOGLE_APPLICATION_CREDENTIALS"]
    optional = ["NCBI_API_KEY", "OPENALEX_EMAIL"]

    all_good = True

    for var in required:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value[:30]}...")
        else:
            print(f"  ❌ {var}: NOT SET")
            all_good = False

    for var in optional:
        value = os.getenv(var)
        if value:
            print(f"  ✅ {var}: {value[:30]}...")
        else:
            print(f"  ⚠️  {var}: Not set (optional)")

    return all_good


def check_credentials():
    """Check if credentials file exists."""
    print("\n" + "=" * 50)
    print("2. Checking Credentials File")
    print("=" * 50)

    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "./credentials.json")

    if os.path.exists(creds_path):
        print(f"  ✅ Credentials file found: {creds_path}")
        return True
    else:
        print(f"  ❌ Credentials file NOT found: {creds_path}")
        print("     Run: gcloud iam service-accounts keys create ./credentials.json ...")
        return False


def check_gcp_connection():
    """Test GCP connection."""
    print("\n" + "=" * 50)
    print("3. Testing GCP Connection")
    print("=" * 50)

    try:
        from google.cloud import bigquery
        from config.gcp_config import config

        client = bigquery.Client(project=config.project_id)
        datasets = list(client.list_datasets())

        print(f"  ✅ Connected to GCP project: {config.project_id}")
        print(f"  ✅ Found {len(datasets)} datasets")

        # Check for our dataset
        dataset_names = [d.dataset_id for d in datasets]
        if "coinvestigator" in dataset_names:
            print("  ✅ 'coinvestigator' dataset exists")
        else:
            print("  ⚠️  'coinvestigator' dataset not found - run bq mk command")

        return True

    except Exception as e:
        print(f"  ❌ GCP connection failed: {e}")
        return False


def check_bigquery_tables():
    """Check if required BigQuery tables exist."""
    print("\n" + "=" * 50)
    print("4. Checking BigQuery Tables")
    print("=" * 50)

    try:
        from google.cloud import bigquery
        from config.gcp_config import config

        client = bigquery.Client(project=config.project_id)

        tables_to_check = [
            "coinvestigator.clingen_gene_disease",
            "coinvestigator.civic_evidence",
            "coinvestigator.reactome_pathways",
            "coinvestigator.string_protein_links",
        ]

        all_exist = True
        for table_ref in tables_to_check:
            try:
                full_ref = f"{config.project_id}.{table_ref}"
                table = client.get_table(full_ref)
                print(f"  ✅ {table_ref}: {table.num_rows} rows")
            except Exception:
                print(f"  ❌ {table_ref}: NOT FOUND")
                all_exist = False

        return all_exist

    except Exception as e:
        print(f"  ❌ Error checking tables: {e}")
        return False


def check_vertex_ai():
    """Test Vertex AI connection."""
    print("\n" + "=" * 50)
    print("5. Testing Vertex AI")
    print("=" * 50)

    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        from config.gcp_config import config

        vertexai.init(project=config.project_id, location=config.location)
        model = GenerativeModel("gemini-2.5-pro")

        # Simple test
        response = model.generate_content("Say 'Hello, Co-Investigator!' in exactly those words.")
        print(f"  ✅ Vertex AI connected")
        print(f"  ✅ Model response: {response.text[:50]}...")
        return True

    except Exception as e:
        print(f"  ❌ Vertex AI test failed: {e}")
        return False


def check_external_apis():
    """Test external API connections."""
    print("\n" + "=" * 50)
    print("6. Testing External APIs")
    print("=" * 50)

    # OpenAlex (no auth required)
    try:
        from tools.search_openalex import search_works
        result = search_works(query="cancer", from_year=2024, per_page=1)
        if result.get("success"):
            print(f"  ✅ OpenAlex API working: {result.get('count')} results")
        else:
            print(f"  ⚠️  OpenAlex returned error: {result.get('error')}")
    except Exception as e:
        print(f"  ❌ OpenAlex test failed: {e}")

    # PubMed
    try:
        from tools.pubmed_entrez import search_pubmed
        result = search_pubmed("cancer", max_results=1)
        if result.get("success"):
            print(f"  ✅ PubMed API working: {result.get('count')} results")
        else:
            print(f"  ⚠️  PubMed returned error: {result.get('error')}")
    except Exception as e:
        print(f"  ❌ PubMed test failed: {e}")

    return True


def main():
    """Run all checks."""
    print("\n🔬 Co-Investigator Setup Verification\n")

    # Load environment
    from dotenv import load_dotenv
    load_dotenv()

    results = []

    results.append(("Environment Variables", check_env()))
    results.append(("Credentials File", check_credentials()))
    results.append(("GCP Connection", check_gcp_connection()))
    results.append(("BigQuery Tables", check_bigquery_tables()))
    results.append(("Vertex AI", check_vertex_ai()))
    results.append(("External APIs", check_external_apis()))

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! Ready to run the app.")
        print("\nRun: streamlit run app/main.py")
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
