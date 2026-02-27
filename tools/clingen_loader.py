"""
ClinGen Data Loader for Co-Investigator Agent.
Loads gene-disease validity data from GCS: gs://benchspark-data-1771447466-datasets/clingen/
"""
from io import StringIO
from typing import Optional

import pandas as pd

from tools.gcs_data_loader import gcs_loader


# Standard ClinGen schema
CLINGEN_SCHEMA = [
    "Gene_Symbol",
    "Gene_ID_HGNC",
    "Disease_Label",
    "Disease_ID_MONDO",
    "MOI",
    "SOP",
    "Classification",
    "Online_Report",
    "Classification_Date",
    "GCEP",
]


class ClinGenLoader:
    """Load and normalize ClinGen gene-disease validity data."""

    PREFIX = "clingen/"

    # Cache for loaded data
    _df_clingen: Optional[pd.DataFrame] = None

    def load_all(self, force_reload: bool = False) -> pd.DataFrame:
        """Load all ClinGen CSV files and combine them."""
        if self._df_clingen is not None and not force_reload:
            return self._df_clingen.copy()

        files = gcs_loader.list_files(self.PREFIX, extension=".csv")
        print(f"   Found {len(files)} ClinGen files")

        df_parts = []
        for filepath in files:
            fname = filepath.split("/")[-1]
            try:
                df = self._load_single_file(filepath, fname)
                if df is not None and not df.empty:
                    df["source_file"] = fname
                    df_parts.append(df)
                    print(f"   ✅ {fname} → {len(df):,} rows")
            except Exception as e:
                print(f"   ⚠️  {fname} skipped: {e}")

        if not df_parts:
            self._df_clingen = pd.DataFrame(columns=CLINGEN_SCHEMA + ["source_file"])
            return self._df_clingen.copy()

        # Combine and deduplicate
        df_combined = pd.concat(df_parts, ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["Gene_Symbol", "Disease_Label"]
        ).reset_index(drop=True)

        # Filter out invalid entries
        df_combined = df_combined[
            ~df_combined["Gene_Symbol"]
            .astype(str)
            .str.upper()
            .isin(["GENE SYMBOL", "GENE", "N/A", "NAN"])
        ]
        df_combined = df_combined[
            df_combined["Gene_Symbol"].str.len() > 1
        ].reset_index(drop=True)

        self._df_clingen = df_combined
        print(f"\n   🧬 ClinGen total: {len(df_combined):,} gene-disease entries")

        return self._df_clingen.copy()

    def _load_single_file(self, filepath: str, fname: str) -> Optional[pd.DataFrame]:
        """Load and normalize a single ClinGen CSV file."""
        raw = gcs_loader.load_text(filepath)

        if fname == "gene-disease-validity.csv":
            # Special handling for gene-disease-validity.csv (skip header rows)
            df = pd.read_csv(
                StringIO(raw),
                skiprows=5,
                names=CLINGEN_SCHEMA,
                on_bad_lines="skip",
                low_memory=False,
            )
            df = df.dropna(subset=["Gene_Symbol"])
            # Filter out multi-gene entries (contain "+")
            df = df[~df["Gene_Symbol"].str.contains(r"\+", na=False)]
        else:
            # Generic handling for other ClinGen files
            df = pd.read_csv(
                StringIO(raw),
                on_bad_lines="skip",
                low_memory=False,
                header=0,
            )
            df.columns = [str(c).strip() for c in df.columns]
            df = self._normalize_columns(df)
            df = df.dropna(subset=["Gene_Symbol"])

        return df

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to standard schema."""
        col_map = {}
        for col in df.columns:
            cu = col.upper()
            if "GENE" in cu and "SYMBOL" in cu:
                col_map[col] = "Gene_Symbol"
            elif "DISEASE" in cu and "LABEL" in cu:
                col_map[col] = "Disease_Label"
            elif "DISEASE" in cu or "CONDITION" in cu:
                col_map[col] = "Disease_Label"
            elif "CLASSIF" in cu or "STATUS" in cu:
                col_map[col] = "Classification"
            elif "HAPLOINSUF" in cu or "TRIPLOS" in cu:
                col_map[col] = "Classification"
            elif "MOI" in cu:
                col_map[col] = "MOI"
            elif "MONDO" in cu:
                col_map[col] = "Disease_ID_MONDO"

        df = df.rename(columns=col_map)

        # Add missing columns
        for col in CLINGEN_SCHEMA:
            if col not in df.columns:
                df[col] = "N/A"

        # Reorder columns
        return df.reindex(columns=CLINGEN_SCHEMA, fill_value="N/A")

    def get_by_classification(
        self, classification: str
    ) -> pd.DataFrame:
        """Get entries with a specific classification."""
        df = self.load_all()
        return df[df["Classification"] == classification].copy()

    def get_definitive_genes(self) -> pd.DataFrame:
        """Get only definitive gene-disease associations."""
        return self.get_by_classification("Definitive")

    def get_strong_genes(self) -> pd.DataFrame:
        """Get only strong gene-disease associations."""
        return self.get_by_classification("Strong")

    def search_by_gene(self, gene_symbol: str) -> pd.DataFrame:
        """Search for entries by gene symbol."""
        df = self.load_all()
        mask = df["Gene_Symbol"].str.upper() == gene_symbol.upper()
        return df[mask].copy()

    def search_by_disease(self, disease_term: str) -> pd.DataFrame:
        """Search for entries by disease term (case-insensitive partial match)."""
        df = self.load_all()
        term_lower = disease_term.lower()
        mask = df["Disease_Label"].str.lower().str.contains(term_lower, na=False)
        return df[mask].copy()


# Global instance
clingen_loader = ClinGenLoader()


def load_clingen() -> pd.DataFrame:
    """Convenience function to load all ClinGen data."""
    return clingen_loader.load_all()
