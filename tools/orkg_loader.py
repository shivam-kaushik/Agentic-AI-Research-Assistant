"""
ORKG (Open Research Knowledge Graph) Data Loader for Co-Investigator Agent.
Loads RDF triples from GCS: gs://benchspark-data-1771447466-datasets/orkg/
"""
import re
from typing import Optional

import pandas as pd

from tools.gcs_data_loader import gcs_loader


# Predicates that contain human-readable label text
LABEL_PREDICATES = {
    "rdfs#label",
    "schema#name",
    "dc#title",
    "dcterms#title",
    "schema#description",
}


class ORKGLoader:
    """Load and parse ORKG N-Triples RDF dump for searchable label text."""

    PREFIX = "orkg/"
    DUMP_FILE = "orkg/orkg-dump.nt"
    MAX_LINES = 500000  # Limit for performance

    # Cache for loaded data
    _df_orkg: Optional[pd.DataFrame] = None

    def load_all(self, force_reload: bool = False) -> pd.DataFrame:
        """Load ORKG label triples from the N-Triples dump."""
        if self._df_orkg is not None and not force_reload:
            return self._df_orkg.copy()

        print(f"   Loading ORKG (label triples — searchable)...")

        rows = []
        lines_processed = 0

        try:
            # Try streaming first (more memory efficient)
            for line in gcs_loader.stream_lines(self.DUMP_FILE, self.MAX_LINES):
                lines_processed += 1
                parsed = self._parse_triple(line)
                if parsed:
                    rows.append(parsed)

            print(
                f"   ✅ ORKG: {len(rows):,} searchable label triples "
                f"(from {lines_processed:,} lines scanned)"
            )

        except Exception as e:
            print(f"   ⚠️  Streaming failed ({e}), trying download...")
            try:
                raw_text = gcs_loader.load_text(self.DUMP_FILE)
                for line in raw_text.split("\n")[: self.MAX_LINES]:
                    lines_processed += 1
                    parsed = self._parse_triple(line)
                    if parsed:
                        rows.append(parsed)

                print(f"   ✅ ORKG fallback: {len(rows):,} label triples")

            except Exception as e2:
                print(f"   ❌ ORKG failed entirely: {e2}")

        if not rows:
            self._df_orkg = pd.DataFrame(columns=["subject", "predicate", "object"])
            return self._df_orkg.copy()

        self._df_orkg = pd.DataFrame(rows)
        return self._df_orkg.copy()

    def _parse_triple(self, line: str) -> Optional[dict]:
        """Parse a single N-Triples line and extract label triples."""
        line = line.strip()
        if not line or line.startswith("#"):
            return None

        # Split into subject, predicate, object
        parts = line.rstrip(" .").split(" ", 2)
        if len(parts) < 3:
            return None

        predicate = parts[1].strip("<>")

        # Only keep label predicates (human-readable text)
        if not self._is_label_predicate(predicate):
            return None

        # Clean and validate the object text
        obj_text = self._clean_rdf_text(parts[2].strip())
        if len(obj_text) < 15 or obj_text.startswith("http"):
            return None

        return {
            "subject": parts[0].strip("<>")[:120],
            "predicate": predicate[:100],
            "object": obj_text[:300],
        }

    def _is_label_predicate(self, predicate: str) -> bool:
        """Check if the predicate is a label predicate."""
        p = predicate.lower()
        if any(k in p for k in LABEL_PREDICATES):
            return True
        if p.endswith("#label") or p.endswith("/label"):
            return True
        return False

    def _clean_rdf_text(self, raw: str) -> str:
        """Clean RDF literal text."""
        # Remove type annotations (^^<...>)
        raw = re.sub(r"\^\^<[^>]+>", "", raw)
        # Remove language tags (@en, @de, etc.)
        raw = re.sub(r"@[a-z]{2,5}$", "", raw)
        # Remove quotes
        return raw.strip().strip('"').strip("'").strip()

    def search_by_object(self, query: str) -> pd.DataFrame:
        """Search the object field (human-readable text)."""
        df = self.load_all()
        query_lower = query.lower()
        mask = df["object"].str.lower().str.contains(query_lower, na=False)
        return df[mask].copy()

    def search_by_subject(self, query: str) -> pd.DataFrame:
        """Search the subject field (URIs)."""
        df = self.load_all()
        query_lower = query.lower()
        mask = df["subject"].str.lower().str.contains(query_lower, na=False)
        return df[mask].copy()

    def multi_search(
        self,
        disease_variants: list[str],
        topic_keywords: list[str],
        gene_variants: list[str],
    ) -> pd.DataFrame:
        """
        Multi-strategy ORKG search on human-readable label triples.
        Searches by disease variants, topic keywords, and gene variants.
        """
        df = self.load_all()
        if df.empty:
            return df

        results = pd.DataFrame(columns=df.columns)

        # Search by disease variants
        for variant in disease_variants:
            if len(variant) >= 8:
                matches = self.search_by_object(variant)
                results = pd.concat([results, matches]).drop_duplicates()

        # Search by topic keywords
        for keyword in topic_keywords:
            if len(keyword) >= 6:
                matches = self.search_by_object(keyword)
                results = pd.concat([results, matches]).drop_duplicates()

        # Search by disease fragments (individual words)
        skip_words = {
            "disease", "disorder", "syndrome", "condition", "related",
            "associated", "induced", "caused", "idiopathic", "familial",
            "congenital", "acquired", "primary", "secondary", "chronic", "acute"
        }
        for variant in disease_variants:
            for word in variant.lower().split():
                if len(word) >= 6 and word not in skip_words:
                    matches = self.search_by_object(word)
                    results = pd.concat([results, matches]).drop_duplicates()

        # Search by gene variants (stricter threshold)
        for gene in gene_variants:
            if len(gene) >= 3:
                matches = self.search_by_object(gene)
                results = pd.concat([results, matches]).drop_duplicates()

        return results.reset_index(drop=True)


# Global instance
orkg_loader = ORKGLoader()


def load_orkg() -> pd.DataFrame:
    """Convenience function to load all ORKG data."""
    return orkg_loader.load_all()
