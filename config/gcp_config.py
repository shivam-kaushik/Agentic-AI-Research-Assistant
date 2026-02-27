"""
GCP Configuration for Co-Investigator Agent
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GCPConfig:
    """Google Cloud Platform configuration."""

    project_id: str = "queryquest-1771952465"
    location: str = "us-central1"

    # BigQuery
    bq_dataset: str = "coinvestigator"

    # Firestore
    firestore_collection_sessions: str = "agent_sessions"
    firestore_collection_hitl: str = "hitl_checkpoints"

    # Vertex AI Models
    planner_model: str = "gemini-2.5-pro"
    synthesizer_model: str = "gemini-2.5-pro"

    # GCS Bucket for source data
    data_bucket: str = "gs://benchspark-data-1771447466-datasets"

    @classmethod
    def from_env(cls) -> "GCPConfig":
        """Load configuration from environment variables."""
        return cls(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT", cls.project_id),
            location=os.getenv("GCP_LOCATION", cls.location),
            bq_dataset=os.getenv("BQ_DATASET", cls.bq_dataset),
        )


@dataclass
class BigQueryTables:
    """BigQuery table references (ClinGen only for hybrid approach)."""

    # ClinGen tables (also available via GCS)
    clingen_gene_disease: str = "coinvestigator.clingen_gene_disease"
    clingen_variant_pathogenicity: str = "coinvestigator.clingen_variant_pathogenicity"


@dataclass
class GCSPaths:
    """GCS paths for datasets (QueryQuest style)."""

    bucket: str = "benchspark-data-1771447466-datasets"

    # Dataset prefixes
    clingen_prefix: str = "clingen/"
    pubmedqa_prefix: str = "pubmedqa/"
    biorxiv_prefix: str = "biorxiv-medrxiv/"
    orkg_prefix: str = "orkg/"
    orkg_dump: str = "orkg/orkg-dump.nt"


# Global instances
config = GCPConfig.from_env()
tables = BigQueryTables()
gcs_paths = GCSPaths()
