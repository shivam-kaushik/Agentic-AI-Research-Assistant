"""
PubMed Entrez API Client for Co-Investigator Agent

Provides access to PubMed abstracts and search via NCBI E-utilities.
https://www.ncbi.nlm.nih.gov/books/NBK25497/
"""
import os
import requests
import xml.etree.ElementTree as ET
from typing import Any
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

ENTREZ_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class PubMedClient:
    """Client for PubMed/NCBI Entrez API."""

    def __init__(self, api_key: str | None = None, email: str | None = None):
        """
        Initialize PubMed client.

        Args:
            api_key: NCBI API key for higher rate limits
            email: Contact email (required by NCBI)
        """
        self.api_key = api_key or os.getenv("NCBI_API_KEY")
        self.email = email or os.getenv("OPENALEX_EMAIL", "coinvestigator@example.com")
        self.base_url = ENTREZ_BASE_URL
        self.session = requests.Session()

    def _get_params(self, **kwargs) -> dict:
        """Build request parameters with authentication."""
        params = {
            "tool": "CoInvestigator",
            "email": self.email,
            **kwargs,
        }
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _make_request(self, endpoint: str, params: dict) -> requests.Response:
        """Make a request to Entrez API with retry logic."""
        url = f"{self.base_url}/{endpoint}"
        response = self.session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response

    def search(
        self,
        query: str,
        max_results: int = 25,
        sort: str = "relevance",
        min_date: str | None = None,
        max_date: str | None = None,
    ) -> dict[str, Any]:
        """
        Search PubMed for articles.

        Args:
            query: Search query (supports PubMed query syntax)
            max_results: Maximum number of results to return
            sort: Sort order (relevance, pub_date)
            min_date: Minimum publication date (YYYY/MM/DD)
            max_date: Maximum publication date (YYYY/MM/DD)

        Returns:
            Dictionary with PMIDs and search metadata
        """
        params = self._get_params(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort=sort,
            retmode="json",
        )

        if min_date:
            params["mindate"] = min_date
            params["datetype"] = "pdat"

        if max_date:
            params["maxdate"] = max_date
            params["datetype"] = "pdat"

        try:
            response = self._make_request("esearch.fcgi", params)
            data = response.json()

            result = data.get("esearchresult", {})
            pmids = result.get("idlist", [])

            return {
                "success": True,
                "source": "pubmed_search",
                "query": query,
                "count": int(result.get("count", 0)),
                "pmids": pmids,
            }
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return {
                "success": False,
                "source": "pubmed_search",
                "query": query,
                "error": str(e),
                "pmids": [],
            }

    def fetch_abstracts(self, pmids: list[str]) -> dict[str, Any]:
        """
        Fetch abstracts for given PMIDs.

        Args:
            pmids: List of PubMed IDs

        Returns:
            Dictionary with article details including abstracts
        """
        if not pmids:
            return {
                "success": True,
                "source": "pubmed_fetch",
                "count": 0,
                "data": [],
            }

        params = self._get_params(
            db="pubmed",
            id=",".join(pmids),
            retmode="xml",
        )

        try:
            response = self._make_request("efetch.fcgi", params)
            articles = self._parse_pubmed_xml(response.text)

            return {
                "success": True,
                "source": "pubmed_fetch",
                "count": len(articles),
                "data": articles,
            }
        except Exception as e:
            logger.error(f"PubMed fetch failed: {e}")
            return {
                "success": False,
                "source": "pubmed_fetch",
                "error": str(e),
                "data": [],
            }

    def search_and_fetch(
        self,
        query: str,
        max_results: int = 10,
        **kwargs
    ) -> dict[str, Any]:
        """
        Search PubMed and fetch abstracts in one call.

        Args:
            query: Search query
            max_results: Maximum results to fetch
            **kwargs: Additional search parameters

        Returns:
            Dictionary with full article details
        """
        # First search
        search_result = self.search(query, max_results=max_results, **kwargs)

        if not search_result.get("success") or not search_result.get("pmids"):
            return {
                "success": search_result.get("success", False),
                "source": "pubmed",
                "query": query,
                "total_found": search_result.get("count", 0),
                "error": search_result.get("error"),
                "data": [],
            }

        # Then fetch abstracts
        fetch_result = self.fetch_abstracts(search_result["pmids"])

        return {
            "success": fetch_result.get("success", False),
            "source": "pubmed",
            "query": query,
            "total_found": search_result.get("count", 0),
            "returned": len(fetch_result.get("data", [])),
            "data": fetch_result.get("data", []),
        }

    def _parse_pubmed_xml(self, xml_text: str) -> list[dict]:
        """Parse PubMed XML response into structured data."""
        articles = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                medline = article.find(".//MedlineCitation")
                if medline is None:
                    continue

                pmid_elem = medline.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else None

                article_elem = medline.find(".//Article")
                if article_elem is None:
                    continue

                # Title
                title_elem = article_elem.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else None

                # Abstract
                abstract_elem = article_elem.find(".//Abstract/AbstractText")
                abstract = abstract_elem.text if abstract_elem is not None else None

                # Handle structured abstracts
                if abstract is None:
                    abstract_parts = article_elem.findall(".//Abstract/AbstractText")
                    if abstract_parts:
                        abstract = " ".join(
                            f"{p.get('Label', '')}: {p.text or ''}"
                            for p in abstract_parts
                            if p.text
                        )

                # Authors
                authors = []
                for author in article_elem.findall(".//AuthorList/Author"):
                    lastname = author.find("LastName")
                    forename = author.find("ForeName")
                    if lastname is not None:
                        name = lastname.text
                        if forename is not None:
                            name = f"{forename.text} {name}"
                        authors.append(name)

                # Journal
                journal_elem = article_elem.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else None

                # Publication date
                pub_date = None
                date_elem = article_elem.find(".//Journal/JournalIssue/PubDate")
                if date_elem is not None:
                    year = date_elem.find("Year")
                    month = date_elem.find("Month")
                    if year is not None:
                        pub_date = year.text
                        if month is not None:
                            pub_date = f"{month.text} {pub_date}"

                # Keywords/MeSH terms
                keywords = []
                for mesh in medline.findall(".//MeshHeadingList/MeshHeading/DescriptorName"):
                    if mesh.text:
                        keywords.append(mesh.text)

                # DOI
                doi = None
                for id_elem in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
                    if id_elem.get("IdType") == "doi":
                        doi = id_elem.text
                        break

                articles.append({
                    "pmid": pmid,
                    "doi": doi,
                    "title": title,
                    "abstract": abstract,
                    "authors": authors[:10],  # Limit to 10 authors
                    "journal": journal,
                    "publication_date": pub_date,
                    "mesh_terms": keywords[:10],  # Limit to 10 terms
                })

        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML: {e}")

        return articles


# Module-level convenience functions
_client: PubMedClient | None = None


def get_pubmed_client() -> PubMedClient:
    """Get or create PubMed client singleton."""
    global _client
    if _client is None:
        _client = PubMedClient()
    return _client


def search_pubmed(query: str, max_results: int = 25, **kwargs) -> dict[str, Any]:
    """Search PubMed for articles matching query."""
    client = get_pubmed_client()
    return client.search(query, max_results=max_results, **kwargs)


def fetch_abstracts(pmids: list[str]) -> dict[str, Any]:
    """Fetch abstracts for given PubMed IDs."""
    client = get_pubmed_client()
    return client.fetch_abstracts(pmids)


def search_and_fetch_pubmed(query: str, max_results: int = 10, **kwargs) -> dict[str, Any]:
    """Search PubMed and fetch abstracts in one call."""
    client = get_pubmed_client()
    return client.search_and_fetch(query, max_results=max_results, **kwargs)
