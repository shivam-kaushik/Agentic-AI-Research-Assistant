"""
PubMedQA Data Loader for Co-Investigator Agent.
Loads Q&A data from GCS: gs://benchspark-data-1771447466-datasets/pubmedqa/
"""
from typing import Optional

import pandas as pd

from tools.gcs_data_loader import gcs_loader


class PubMedQALoader:
    """Load and process PubMedQA question-answer data."""

    PREFIX = "pubmedqa/"

    # Cache for loaded data
    _df_all: Optional[pd.DataFrame] = None
    _df_searchable: Optional[pd.DataFrame] = None

    def load_all(self, force_reload: bool = False) -> pd.DataFrame:
        """Load all PubMedQA JSON files."""
        if self._df_all is not None and not force_reload:
            return self._df_all.copy()

        files = gcs_loader.list_files(self.PREFIX, extension=".json")
        print(f"   Found {len(files)} PubMedQA files")

        rows = []
        for filepath in files:
            fname = filepath.split("/")[-1]
            try:
                data = gcs_loader.load_json(filepath)
                file_rows = self._parse_file(data, filepath, fname)
                rows.extend(file_rows)
                print(f"   ✅ {fname} → {len(data):,} entries")
            except Exception as e:
                print(f"   ⚠️  {fname} skipped: {e}")

        if not rows:
            self._df_all = pd.DataFrame(
                columns=["ID", "Question", "Answer", "Context", "Type", "source_file"]
            )
            return self._df_all.copy()

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["ID"]).reset_index(drop=True)

        self._df_all = df
        self._df_searchable = None  # Reset searchable cache

        return self._df_all.copy()

    def _parse_file(
        self, data: dict, filepath: str, fname: str
    ) -> list[dict]:
        """Parse a single PubMedQA JSON file."""
        rows = []

        if "pqal" in filepath.lower():
            # Labelled dataset with questions and answers
            for key, entry in data.items():
                question = entry.get("QUESTION", "")
                contexts = entry.get("CONTEXTS", [])
                answer = entry.get("final_decision", "").upper()

                rows.append({
                    "ID": str(key),
                    "Question": question,
                    "Answer": answer,
                    "Context": " ".join(contexts)[:500] if contexts else "",
                    "Type": "labelled",
                    "source_file": fname,
                })

        elif "ground_truth" in filepath.lower():
            # Ground truth file (just answers)
            for key, answer in data.items():
                rows.append({
                    "ID": f"gt_{key}",
                    "Question": "",
                    "Answer": str(answer).upper(),
                    "Context": "",
                    "Type": "ground_truth",
                    "source_file": fname,
                })

        else:
            # Try to parse as generic format
            for key, entry in data.items():
                if isinstance(entry, dict):
                    question = entry.get("QUESTION", entry.get("question", ""))
                    answer = entry.get("final_decision", entry.get("answer", ""))
                    contexts = entry.get("CONTEXTS", entry.get("contexts", []))

                    rows.append({
                        "ID": str(key),
                        "Question": question,
                        "Answer": str(answer).upper(),
                        "Context": " ".join(contexts)[:500] if isinstance(contexts, list) else str(contexts)[:500],
                        "Type": "other",
                        "source_file": fname,
                    })

        return rows

    def load_searchable(self, force_reload: bool = False) -> pd.DataFrame:
        """Load only searchable (labelled) entries with valid questions."""
        if self._df_searchable is not None and not force_reload:
            return self._df_searchable.copy()

        df = self.load_all(force_reload=force_reload)

        # Filter for searchable entries
        mask = (df["Type"] == "labelled") & (df["Question"].str.len() > 5)
        self._df_searchable = df[mask].copy().reset_index(drop=True)

        print(
            f"\n   ❓ Searchable: {len(self._df_searchable):,} | "
            f"Ground truth: {len(df) - len(self._df_searchable):,}"
        )

        return self._df_searchable.copy()

    def search_by_question(self, query: str) -> pd.DataFrame:
        """Search questions by keyword (case-insensitive)."""
        df = self.load_searchable()
        query_lower = query.lower()
        mask = df["Question"].str.lower().str.contains(query_lower, na=False)
        return df[mask].copy()

    def search_by_context(self, query: str) -> pd.DataFrame:
        """Search context by keyword (case-insensitive)."""
        df = self.load_searchable()
        query_lower = query.lower()
        mask = df["Context"].str.lower().str.contains(query_lower, na=False)
        return df[mask].copy()

    def get_by_answer(self, answer: str) -> pd.DataFrame:
        """Get entries with a specific answer (YES, NO, MAYBE)."""
        df = self.load_searchable()
        return df[df["Answer"] == answer.upper()].copy()

    def get_yes_answers(self) -> pd.DataFrame:
        """Get entries with YES answers."""
        return self.get_by_answer("YES")

    def get_no_answers(self) -> pd.DataFrame:
        """Get entries with NO answers."""
        return self.get_by_answer("NO")


# Global instance
pubmedqa_loader = PubMedQALoader()


def load_pubmedqa() -> pd.DataFrame:
    """Convenience function to load all searchable PubMedQA data."""
    return pubmedqa_loader.load_searchable()
