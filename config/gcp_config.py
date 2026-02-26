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
    """BigQuery table references."""

    # ClinGen tables
    clingen_gene_disease: str = "coinvestigator.clingen_gene_disease"
    clingen_variant_pathogenicity: str = "coinvestigator.clingen_variant_pathogenicity"

    # CIViC tables
    civic_evidence: str = "coinvestigator.civic_evidence"
    civic_variants: str = "coinvestigator.civic_variants"

    # Reactome tables
    reactome_pathways: str = "coinvestigator.reactome_pathways"
    reactome_pathway_relations: str = "coinvestigator.reactome_pathway_relations"
    reactome_uniprot_mapping: str = "coinvestigator.reactome_uniprot_mapping"

    # STRING tables
    string_protein_links: str = "coinvestigator.string_protein_links"
    string_protein_info: str = "coinvestigator.string_protein_info"


# Global instances
config = GCPConfig.from_env()
tables = BigQueryTables()
