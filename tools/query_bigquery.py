"""
BigQuery Query Tools for Co-Investigator Agent

Provides parameterized SQL queries for biomedical datasets:
- ClinGen: Gene-disease validity
- CIViC: Clinical cancer variant evidence
- Reactome: Biological pathways
- STRING: Protein-protein interactions
"""
from google.cloud import bigquery
from google.cloud.exceptions import GoogleCloudError
import pandas as pd
from typing import Any
import logging

import sys
sys.path.append("..")
from config.gcp_config import config, tables

logger = logging.getLogger(__name__)


class BigQueryClient:
    """Client for executing BigQuery queries."""

    def __init__(self, project_id: str | None = None):
        self.project_id = project_id or config.project_id
        self.client = bigquery.Client(project=self.project_id)

    def execute_query(self, query: str, params: dict | None = None) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        try:
            job_config = bigquery.QueryJobConfig()

            if params:
                job_config.query_parameters = [
                    bigquery.ScalarQueryParameter(name, "STRING", value)
                    for name, value in params.items()
                ]

            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            return results.to_dataframe()

        except GoogleCloudError as e:
            logger.error(f"BigQuery error: {e}")
            raise

    def table_exists(self, table_ref: str) -> bool:
        """Check if a table exists."""
        try:
            full_table_id = f"{self.project_id}.{table_ref}"
            self.client.get_table(full_table_id)
            return True
        except Exception:
            return False


# Singleton client instance
_bq_client: BigQueryClient | None = None


def get_bq_client() -> BigQueryClient:
    """Get or create BigQuery client singleton."""
    global _bq_client
    if _bq_client is None:
        _bq_client = BigQueryClient()
    return _bq_client


def query_clingen(
    disease: str | None = None,
    gene: str | None = None,
    limit: int = 100
) -> dict[str, Any]:
    """
    Query ClinGen gene-disease validity data.

    Args:
        disease: Disease name to search for (partial match)
        gene: Gene symbol to search for
        limit: Maximum number of results

    Returns:
        Dictionary with query results and metadata
    """
    client = get_bq_client()

    conditions = []
    if disease:
        conditions.append(f"LOWER(string_field_2) LIKE LOWER('%{disease}%')")
    if gene:
        conditions.append(f"LOWER(string_field_0) LIKE LOWER('%{gene}%')")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
    SELECT *
    FROM `{config.project_id}.{tables.clingen_gene_disease}`
    WHERE {where_clause}
    LIMIT {limit}
    """

    try:
        df = client.execute_query(query)
        return {
            "success": True,
            "source": "clingen_gene_disease",
            "query": {"disease": disease, "gene": gene},
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"ClinGen query failed: {e}")
        return {
            "success": False,
            "source": "clingen_gene_disease",
            "query": {"disease": disease, "gene": gene},
            "error": str(e),
            "data": [],
        }


def query_clingen_variants(
    gene: str | None = None,
    classification: str | None = None,
    limit: int = 100
) -> dict[str, Any]:
    """
    Query ClinGen variant pathogenicity data.

    Args:
        gene: Gene symbol to search for
        classification: Pathogenicity classification filter
        limit: Maximum number of results

    Returns:
        Dictionary with query results and metadata
    """
    client = get_bq_client()

    conditions = []
    if gene:
        conditions.append(f"LOWER(gene) LIKE LOWER('%{gene}%')")
    if classification:
        conditions.append(f"LOWER(classification) LIKE LOWER('%{classification}%')")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
    SELECT *
    FROM `{config.project_id}.{tables.clingen_variant_pathogenicity}`
    WHERE {where_clause}
    LIMIT {limit}
    """

    try:
        df = client.execute_query(query)
        return {
            "success": True,
            "source": "clingen_variant_pathogenicity",
            "query": {"gene": gene, "classification": classification},
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"ClinGen variants query failed: {e}")
        return {
            "success": False,
            "source": "clingen_variant_pathogenicity",
            "query": {"gene": gene, "classification": classification},
            "error": str(e),
            "data": [],
        }


def query_civic(
    disease: str | None = None,
    gene: str | None = None,
    evidence_level: str | None = None,
    limit: int = 100
) -> dict[str, Any]:
    """
    Query CIViC clinical evidence data.

    Args:
        disease: Disease name to search
        gene: Gene symbol to search
        evidence_level: Filter by evidence level (A, B, C, D, E)
        limit: Maximum number of results

    Returns:
        Dictionary with query results and metadata
    """
    client = get_bq_client()

    conditions = []
    if disease:
        conditions.append(f"LOWER(disease) LIKE LOWER('%{disease}%')")
    if gene:
        conditions.append(f"LOWER(molecular_profile) LIKE LOWER('%{gene}%')")
    if evidence_level:
        conditions.append(f"evidence_level = '{evidence_level}'")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
    SELECT *
    FROM `{config.project_id}.{tables.civic_evidence}`
    WHERE {where_clause}
    LIMIT {limit}
    """

    try:
        df = client.execute_query(query)
        return {
            "success": True,
            "source": "civic_evidence",
            "query": {"disease": disease, "gene": gene, "evidence_level": evidence_level},
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"CIViC query failed: {e}")
        return {
            "success": False,
            "source": "civic_evidence",
            "query": {"disease": disease, "gene": gene, "evidence_level": evidence_level},
            "error": str(e),
            "data": [],
        }


def query_civic_variants(
    gene: str | None = None,
    variant_name: str | None = None,
    limit: int = 100
) -> dict[str, Any]:
    """
    Query CIViC variant summaries.

    Args:
        gene: Gene symbol to search
        variant_name: Variant name to search
        limit: Maximum number of results

    Returns:
        Dictionary with query results and metadata
    """
    client = get_bq_client()

    conditions = []
    if gene:
        conditions.append(f"LOWER(gene) LIKE LOWER('%{gene}%')")
    if variant_name:
        conditions.append(f"LOWER(variant) LIKE LOWER('%{variant_name}%')")

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    query = f"""
    SELECT *
    FROM `{config.project_id}.{tables.civic_variants}`
    WHERE {where_clause}
    LIMIT {limit}
    """

    try:
        df = client.execute_query(query)
        return {
            "success": True,
            "source": "civic_variants",
            "query": {"gene": gene, "variant_name": variant_name},
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"CIViC variants query failed: {e}")
        return {
            "success": False,
            "source": "civic_variants",
            "query": {"gene": gene, "variant_name": variant_name},
            "error": str(e),
            "data": [],
        }


def query_reactome(
    pathway_name: str | None = None,
    uniprot_id: str | None = None,
    species: str = "Homo sapiens",
    limit: int = 100
) -> dict[str, Any]:
    """
    Query Reactome pathway data.

    Args:
        pathway_name: Pathway name to search
        uniprot_id: UniProt protein ID to find pathways for
        species: Species filter (default: Homo sapiens)
        limit: Maximum number of results

    Returns:
        Dictionary with query results and metadata
    """
    client = get_bq_client()

    if uniprot_id:
        # Query UniProt to Reactome mapping
        query = f"""
        SELECT *
        FROM `{config.project_id}.{tables.reactome_uniprot_mapping}`
        WHERE uniprot_id = '{uniprot_id}'
          AND LOWER(species) LIKE LOWER('%{species}%')
        LIMIT {limit}
        """
    else:
        # Query pathways directly
        conditions = [f"LOWER(species) LIKE LOWER('%{species}%')"]
        if pathway_name:
            conditions.append(f"LOWER(pathway_name) LIKE LOWER('%{pathway_name}%')")

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT *
        FROM `{config.project_id}.{tables.reactome_pathways}`
        WHERE {where_clause}
        LIMIT {limit}
        """

    try:
        df = client.execute_query(query)
        return {
            "success": True,
            "source": "reactome",
            "query": {"pathway_name": pathway_name, "uniprot_id": uniprot_id, "species": species},
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"Reactome query failed: {e}")
        return {
            "success": False,
            "source": "reactome",
            "query": {"pathway_name": pathway_name, "uniprot_id": uniprot_id, "species": species},
            "error": str(e),
            "data": [],
        }


def query_string(
    protein: str | None = None,
    min_score: int = 700,
    limit: int = 100
) -> dict[str, Any]:
    """
    Query STRING protein-protein interaction data.

    Args:
        protein: Protein ID to find interactions for
        min_score: Minimum combined score threshold (0-1000)
        limit: Maximum number of results

    Returns:
        Dictionary with query results and metadata
    """
    client = get_bq_client()

    if not protein:
        return {
            "success": False,
            "source": "string",
            "error": "Protein ID is required",
            "data": [],
        }

    query = f"""
    SELECT *
    FROM `{config.project_id}.{tables.string_protein_links}`
    WHERE (protein1 LIKE '%{protein}%' OR protein2 LIKE '%{protein}%')
      AND combined_score >= {min_score}
    ORDER BY combined_score DESC
    LIMIT {limit}
    """

    try:
        df = client.execute_query(query)
        return {
            "success": True,
            "source": "string_interactions",
            "query": {"protein": protein, "min_score": min_score},
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"STRING query failed: {e}")
        return {
            "success": False,
            "source": "string_interactions",
            "query": {"protein": protein, "min_score": min_score},
            "error": str(e),
            "data": [],
        }


def query_string_protein_info(
    protein: str | None = None,
    preferred_name: str | None = None,
    limit: int = 100
) -> dict[str, Any]:
    """
    Query STRING protein information.

    Args:
        protein: Protein ID to search
        preferred_name: Preferred protein name to search
        limit: Maximum number of results

    Returns:
        Dictionary with query results and metadata
    """
    client = get_bq_client()

    conditions = []
    if protein:
        conditions.append(f"`#string_protein_id` LIKE '%{protein}%'")
    if preferred_name:
        conditions.append(f"LOWER(preferred_name) LIKE LOWER('%{preferred_name}%')")

    if not conditions:
        return {
            "success": False,
            "source": "string_protein_info",
            "error": "At least one search parameter is required",
            "data": [],
        }

    where_clause = " OR ".join(conditions)

    query = f"""
    SELECT *
    FROM `{config.project_id}.{tables.string_protein_info}`
    WHERE {where_clause}
    LIMIT {limit}
    """

    try:
        df = client.execute_query(query)
        return {
            "success": True,
            "source": "string_protein_info",
            "query": {"protein": protein, "preferred_name": preferred_name},
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }
    except Exception as e:
        logger.error(f"STRING protein info query failed: {e}")
        return {
            "success": False,
            "source": "string_protein_info",
            "query": {"protein": protein, "preferred_name": preferred_name},
            "error": str(e),
            "data": [],
        }


# Dispatcher function for routing queries based on data source
def execute_bigquery_tool(
    data_source: str,
    query_type: str,
    entities: list[str],
    **kwargs
) -> dict[str, Any]:
    """
    Route and execute BigQuery queries based on data source and query type.

    Args:
        data_source: The data source to query (clingen, civic, reactome, string)
        query_type: The type of query (gene_disease, variant, pathway, interaction)
        entities: List of entities to search for
        **kwargs: Additional query parameters

    Returns:
        Combined results from all entity queries
    """
    results = []

    for entity in entities:
        if data_source == "clingen":
            if query_type == "gene_disease":
                result = query_clingen(disease=entity, **kwargs)
            elif query_type == "variant":
                result = query_clingen_variants(gene=entity, **kwargs)
            else:
                result = query_clingen(gene=entity, **kwargs)

        elif data_source == "civic":
            if query_type == "variant":
                result = query_civic_variants(gene=entity, **kwargs)
            else:
                result = query_civic(gene=entity, **kwargs)

        elif data_source == "reactome":
            if query_type == "pathway":
                result = query_reactome(pathway_name=entity, **kwargs)
            else:
                result = query_reactome(uniprot_id=entity, **kwargs)

        elif data_source == "string":
            if query_type == "interaction":
                result = query_string(protein=entity, **kwargs)
            else:
                result = query_string_protein_info(preferred_name=entity, **kwargs)

        else:
            result = {
                "success": False,
                "error": f"Unknown data source: {data_source}",
                "data": [],
            }

        results.append(result)

    # Combine results
    combined_data = []
    total_count = 0
    errors = []

    for r in results:
        if r.get("success"):
            combined_data.extend(r.get("data", []))
            total_count += r.get("count", 0)
        else:
            errors.append(r.get("error"))

    return {
        "success": len(errors) == 0,
        "source": data_source,
        "query_type": query_type,
        "entities": entities,
        "total_count": total_count,
        "data": combined_data,
        "errors": errors if errors else None,
    }
