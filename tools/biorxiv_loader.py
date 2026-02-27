"""
bioRxiv/medRxiv Data Loader for Co-Investigator Agent.
Loads preprint data from GCS: gs://benchspark-data-1771447466-datasets/biorxiv-medrxiv/
"""
from typing import Optional

import pandas as pd

from tools.gcs_data_loader import gcs_loader


class BioRxivLoader:
    """Load and process bioRxiv/medRxiv preprint data."""

    PREFIX = "biorxiv-medrxiv/"

    # Cache for loaded data
    _df_biorxiv: Optional[pd.DataFrame] = None

    def load_all(self, force_reload: bool = False) -> pd.DataFrame:
        """Load all bioRxiv/medRxiv JSON files."""
        if self._df_biorxiv is not None and not force_reload:
            return self._df_biorxiv.copy()

        files = gcs_loader.list_files(self.PREFIX, extension=".json")
        print(f"   Found {len(files)} bioRxiv/medRxiv files")

        rows = []
        for filepath in files:
            fname = filepath.split("/")[-1]
            try:
                data = gcs_loader.load_json(filepath)
                file_rows = self._parse_file(data, filepath)
                rows.extend(file_rows)
                print(f"   ✅ {fname} → {len(file_rows):,} preprints")
            except Exception as e:
                print(f"   ⚠️  {fname} skipped: {e}")

        if not rows:
            self._df_biorxiv = pd.DataFrame(
                columns=[
                    "Title", "Authors", "Date", "DOI",
                    "Abstract", "Category", "source"
                ]
            )
            return self._df_biorxiv.copy()

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["Title"]).reset_index(drop=True)

        # Convert Date to datetime
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        self._df_biorxiv = df
        print(f"\n   📰 bioRxiv/medRxiv total: {len(df):,} preprints")

        return self._df_biorxiv.copy()

    def _parse_file(self, data: dict, filepath: str) -> list[dict]:
        """Parse a single bioRxiv/medRxiv JSON file."""
        rows = []

        # Determine source (biorxiv or medrxiv)
        source = "medrxiv" if "medrxiv" in filepath.lower() else "biorxiv"

        # Handle "collection" format
        collection = data.get("collection", [])
        if not collection and isinstance(data, list):
            collection = data

        for preprint in collection:
            if not isinstance(preprint, dict):
                continue

            rows.append({
                "Title": str(preprint.get("title", "")).strip(),
                "Authors": str(preprint.get("authors", ""))[:150],
                "Date": str(preprint.get("date", "")),
                "DOI": str(preprint.get("doi", "")),
                "Abstract": str(preprint.get("abstract", ""))[:500],
                "Category": str(preprint.get("category", "")),
                "source": source,
            })

        return rows

    def get_by_source(self, source: str) -> pd.DataFrame:
        """Get preprints from a specific source (biorxiv or medrxiv)."""
        df = self.load_all()
        return df[df["source"] == source.lower()].copy()

    def get_biorxiv(self) -> pd.DataFrame:
        """Get only bioRxiv preprints."""
        return self.get_by_source("biorxiv")

    def get_medrxiv(self) -> pd.DataFrame:
        """Get only medRxiv preprints."""
        return self.get_by_source("medrxiv")

    def search_by_title(self, query: str) -> pd.DataFrame:
        """Search preprints by title (case-insensitive)."""
        df = self.load_all()
        query_lower = query.lower()
        mask = df["Title"].str.lower().str.contains(query_lower, na=False)
        return df[mask].copy()

    def search_by_abstract(self, query: str) -> pd.DataFrame:
        """Search preprints by abstract (case-insensitive)."""
        df = self.load_all()
        query_lower = query.lower()
        mask = df["Abstract"].str.lower().str.contains(query_lower, na=False)
        return df[mask].copy()

    def get_by_category(self, category: str) -> pd.DataFrame:
        """Get preprints by category."""
        df = self.load_all()
        category_lower = category.lower()
        mask = df["Category"].str.lower().str.contains(category_lower, na=False)
        return df[mask].copy()

    def get_recent(self, n: int = 100) -> pd.DataFrame:
        """Get the most recent preprints."""
        df = self.load_all()
        return df.sort_values("Date", ascending=False).head(n).copy()

    def get_by_date_range(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Get preprints within a date range."""
        df = self.load_all()

        if start_date:
            df = df[df["Date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.to_datetime(end_date)]

        return df.copy()


# Global instance
biorxiv_loader = BioRxivLoader()


def load_biorxiv() -> pd.DataFrame:
    """Convenience function to load all bioRxiv/medRxiv data."""
    return biorxiv_loader.load_all()
