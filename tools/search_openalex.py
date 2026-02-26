"""
OpenAlex API Client for Co-Investigator Agent

Provides access to researcher and publication data via the OpenAlex API.
https://docs.openalex.org/
"""
import os
import requests
from typing import Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

OPENALEX_BASE_URL = "https://api.openalex.org"


class OpenAlexClient:
    """Client for OpenAlex API."""

    def __init__(self, email: str | None = None):
        """
        Initialize OpenAlex client.

        Args:
            email: Contact email for polite pool (faster rate limits)
        """
        self.email = email or os.getenv("OPENALEX_EMAIL")
        self.base_url = OPENALEX_BASE_URL
        self.session = requests.Session()

        # Set up headers for polite pool
        if self.email:
            self.session.headers["User-Agent"] = f"CoInvestigator/1.0 (mailto:{self.email})"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, endpoint: str, params: dict | None = None) -> dict:
        """Make a request to OpenAlex API with retry logic."""
        url = f"{self.base_url}/{endpoint}"

        if params is None:
            params = {}

        # Add email for polite pool
        if self.email:
            params["mailto"] = self.email

        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def search_works(
        self,
        query: str | None = None,
        doi: str | None = None,
        pmid: str | None = None,
        title: str | None = None,
        concept: str | None = None,
        from_year: int | None = None,
        per_page: int = 25
    ) -> dict[str, Any]:
        """
        Search for works (publications) in OpenAlex.

        Args:
            query: Free text search query
            doi: Search by DOI
            pmid: Search by PubMed ID
            title: Search by title
            concept: Filter by concept/topic
            from_year: Filter works from this year onwards
            per_page: Number of results per page

        Returns:
            Dictionary with search results
        """
        params = {"per_page": per_page}
        filters = []

        if query:
            params["search"] = query

        if doi:
            filters.append(f"doi:{doi}")

        if pmid:
            filters.append(f"pmid:{pmid}")

        if title:
            params["filter"] = f"title.search:{title}"

        if concept:
            filters.append(f"concepts.display_name.search:{concept}")

        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")

        if filters:
            params["filter"] = ",".join(filters)

        try:
            data = self._make_request("works", params)
            return {
                "success": True,
                "source": "openalex_works",
                "query": {
                    "query": query,
                    "doi": doi,
                    "pmid": pmid,
                    "concept": concept,
                    "from_year": from_year,
                },
                "count": data.get("meta", {}).get("count", 0),
                "data": self._format_works(data.get("results", [])),
            }
        except Exception as e:
            logger.error(f"OpenAlex works search failed: {e}")
            return {
                "success": False,
                "source": "openalex_works",
                "error": str(e),
                "data": [],
            }

    def search_authors(
        self,
        query: str | None = None,
        orcid: str | None = None,
        institution: str | None = None,
        concept: str | None = None,
        per_page: int = 25
    ) -> dict[str, Any]:
        """
        Search for authors/researchers in OpenAlex.

        Args:
            query: Free text search for author name
            orcid: Search by ORCID
            institution: Filter by institution name
            concept: Filter by research concept/topic
            per_page: Number of results per page

        Returns:
            Dictionary with search results
        """
        params = {"per_page": per_page}
        filters = []

        if query:
            params["search"] = query

        if orcid:
            filters.append(f"orcid:{orcid}")

        if institution:
            filters.append(f"affiliations.institution.display_name.search:{institution}")

        if concept:
            filters.append(f"x_concepts.display_name.search:{concept}")

        if filters:
            params["filter"] = ",".join(filters)

        try:
            data = self._make_request("authors", params)
            return {
                "success": True,
                "source": "openalex_authors",
                "query": {
                    "query": query,
                    "orcid": orcid,
                    "institution": institution,
                    "concept": concept,
                },
                "count": data.get("meta", {}).get("count", 0),
                "data": self._format_authors(data.get("results", [])),
            }
        except Exception as e:
            logger.error(f"OpenAlex authors search failed: {e}")
            return {
                "success": False,
                "source": "openalex_authors",
                "error": str(e),
                "data": [],
            }

    def get_author_works(
        self,
        author_id: str,
        from_year: int | None = None,
        per_page: int = 25
    ) -> dict[str, Any]:
        """
        Get works by a specific author.

        Args:
            author_id: OpenAlex author ID
            from_year: Filter works from this year onwards
            per_page: Number of results per page

        Returns:
            Dictionary with author's works
        """
        params = {"per_page": per_page}
        filters = [f"author.id:{author_id}"]

        if from_year:
            filters.append(f"from_publication_date:{from_year}-01-01")

        params["filter"] = ",".join(filters)
        params["sort"] = "publication_date:desc"

        try:
            data = self._make_request("works", params)
            return {
                "success": True,
                "source": "openalex_author_works",
                "query": {"author_id": author_id, "from_year": from_year},
                "count": data.get("meta", {}).get("count", 0),
                "data": self._format_works(data.get("results", [])),
            }
        except Exception as e:
            logger.error(f"OpenAlex author works failed: {e}")
            return {
                "success": False,
                "source": "openalex_author_works",
                "error": str(e),
                "data": [],
            }

    def search_institutions(
        self,
        query: str,
        country: str | None = None,
        per_page: int = 25
    ) -> dict[str, Any]:
        """
        Search for research institutions.

        Args:
            query: Institution name search
            country: Filter by country code
            per_page: Number of results per page

        Returns:
            Dictionary with institution results
        """
        params = {"search": query, "per_page": per_page}

        if country:
            params["filter"] = f"country_code:{country}"

        try:
            data = self._make_request("institutions", params)
            return {
                "success": True,
                "source": "openalex_institutions",
                "query": {"query": query, "country": country},
                "count": data.get("meta", {}).get("count", 0),
                "data": self._format_institutions(data.get("results", [])),
            }
        except Exception as e:
            logger.error(f"OpenAlex institutions search failed: {e}")
            return {
                "success": False,
                "source": "openalex_institutions",
                "error": str(e),
                "data": [],
            }

    def _format_works(self, works: list) -> list[dict]:
        """Format works results for consistent output."""
        formatted = []
        for work in works:
            formatted.append({
                "id": work.get("id"),
                "doi": work.get("doi"),
                "title": work.get("title"),
                "publication_date": work.get("publication_date"),
                "publication_year": work.get("publication_year"),
                "cited_by_count": work.get("cited_by_count"),
                "type": work.get("type"),
                "open_access": work.get("open_access", {}).get("is_oa"),
                "authors": [
                    {
                        "name": a.get("author", {}).get("display_name"),
                        "id": a.get("author", {}).get("id"),
                        "institution": a.get("institutions", [{}])[0].get("display_name")
                        if a.get("institutions") else None,
                    }
                    for a in work.get("authorships", [])[:5]  # Limit to first 5 authors
                ],
                "concepts": [
                    {"name": c.get("display_name"), "score": c.get("score")}
                    for c in work.get("concepts", [])[:5]  # Limit to top 5 concepts
                ],
                "abstract": work.get("abstract_inverted_index") is not None,
            })
        return formatted

    def _format_authors(self, authors: list) -> list[dict]:
        """Format author results for consistent output."""
        formatted = []
        for author in authors:
            formatted.append({
                "id": author.get("id"),
                "orcid": author.get("orcid"),
                "display_name": author.get("display_name"),
                "works_count": author.get("works_count"),
                "cited_by_count": author.get("cited_by_count"),
                "h_index": author.get("summary_stats", {}).get("h_index"),
                "last_known_institution": author.get("last_known_institution", {}).get("display_name"),
                "top_concepts": [
                    {"name": c.get("display_name"), "score": c.get("score")}
                    for c in author.get("x_concepts", [])[:5]
                ],
            })
        return formatted

    def _format_institutions(self, institutions: list) -> list[dict]:
        """Format institution results for consistent output."""
        formatted = []
        for inst in institutions:
            formatted.append({
                "id": inst.get("id"),
                "display_name": inst.get("display_name"),
                "country_code": inst.get("country_code"),
                "type": inst.get("type"),
                "works_count": inst.get("works_count"),
                "cited_by_count": inst.get("cited_by_count"),
                "homepage_url": inst.get("homepage_url"),
            })
        return formatted


# Module-level convenience functions
_client: OpenAlexClient | None = None


def get_openalex_client() -> OpenAlexClient:
    """Get or create OpenAlex client singleton."""
    global _client
    if _client is None:
        _client = OpenAlexClient()
    return _client


def search_researchers(
    query: str | None = None,
    concept: str | None = None,
    institution: str | None = None,
    **kwargs
) -> dict[str, Any]:
    """Search for researchers by name, concept, or institution."""
    client = get_openalex_client()
    return client.search_authors(
        query=query,
        concept=concept,
        institution=institution,
        **kwargs
    )


def search_works(
    query: str | None = None,
    concept: str | None = None,
    from_year: int | None = None,
    **kwargs
) -> dict[str, Any]:
    """Search for publications by query or concept."""
    client = get_openalex_client()
    return client.search_works(
        query=query,
        concept=concept,
        from_year=from_year,
        **kwargs
    )
