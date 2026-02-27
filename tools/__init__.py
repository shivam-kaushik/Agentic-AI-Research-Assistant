"""Tools module for Co-Investigator Agent."""
# BigQuery (ClinGen only)
from .query_bigquery import BigQueryClient, query_clingen

# External APIs
from .search_openalex import OpenAlexClient, search_researchers, search_works
from .pubmed_entrez import PubMedClient, search_pubmed, fetch_abstracts

# GCS Data Loaders
from .gcs_data_loader import GCSDataLoader, gcs_loader
from .clingen_loader import ClinGenLoader, clingen_loader, load_clingen
from .pubmedqa_loader import PubMedQALoader, pubmedqa_loader, load_pubmedqa
from .biorxiv_loader import BioRxivLoader, biorxiv_loader, load_biorxiv
from .orkg_loader import ORKGLoader, orkg_loader, load_orkg

# Search Utilities
from .search_utils import (
    smart_search,
    gemini_filter,
    safe_len,
    combine_search_results,
    search_all_datasets,
    summarize_search_results,
)

__all__ = [
    # BigQuery
    "BigQueryClient",
    "query_clingen",
    # External APIs
    "OpenAlexClient",
    "search_researchers",
    "search_works",
    "PubMedClient",
    "search_pubmed",
    "fetch_abstracts",
    # GCS Loaders
    "GCSDataLoader",
    "gcs_loader",
    "ClinGenLoader",
    "clingen_loader",
    "load_clingen",
    "PubMedQALoader",
    "pubmedqa_loader",
    "load_pubmedqa",
    "BioRxivLoader",
    "biorxiv_loader",
    "load_biorxiv",
    "ORKGLoader",
    "orkg_loader",
    "load_orkg",
    # Search Utilities
    "smart_search",
    "gemini_filter",
    "safe_len",
    "combine_search_results",
    "search_all_datasets",
    "summarize_search_results",
]
