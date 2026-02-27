"""
GCS Data Loader for Co-Investigator Agent.
Loads datasets from the GCS bucket: gs://benchspark-data-1771447466-datasets/
"""
import json
import re
from io import StringIO
from typing import Optional

import pandas as pd
from google.cloud import storage

from config.gcp_config import config


class GCSDataLoader:
    """Base class for loading datasets from GCS bucket."""

    BUCKET_NAME = "benchspark-data-1771447466-datasets"

    # In-memory cache to avoid repeated downloads
    _cache: dict = {}

    def __init__(self):
        self.client = storage.Client(project=config.project_id)
        self.bucket = self.client.bucket(self.BUCKET_NAME)

    def list_files(self, prefix: str, extension: Optional[str] = None) -> list[str]:
        """List files in a GCS prefix."""
        blobs = self.client.list_blobs(self.BUCKET_NAME, prefix=prefix)
        files = [b.name for b in blobs if not b.name.endswith("/")]
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return sorted(files)

    def load_text(self, filepath: str, use_cache: bool = True) -> str:
        """Load text content from a GCS file."""
        cache_key = f"text:{filepath}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        blob = self.bucket.blob(filepath)
        content = blob.download_as_text(encoding="utf-8")

        if use_cache:
            self._cache[cache_key] = content

        return content

    def load_json(self, filepath: str, use_cache: bool = True) -> dict:
        """Load JSON content from a GCS file."""
        cache_key = f"json:{filepath}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        content = self.load_text(filepath, use_cache=False)
        data = json.loads(content)

        if use_cache:
            self._cache[cache_key] = data

        return data

    def load_csv(
        self,
        filepath: str,
        use_cache: bool = True,
        **pandas_kwargs
    ) -> pd.DataFrame:
        """Load CSV content from a GCS file."""
        cache_key = f"csv:{filepath}:{str(pandas_kwargs)}"
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key].copy()

        content = self.load_text(filepath, use_cache=False)
        df = pd.read_csv(StringIO(content), **pandas_kwargs)

        if use_cache:
            self._cache[cache_key] = df

        return df.copy()

    def stream_lines(self, filepath: str, max_lines: int = 500000):
        """Stream lines from a large GCS file."""
        blob = self.bucket.blob(filepath)
        count = 0
        with blob.open("r", encoding="utf-8") as f:
            for line in f:
                if count >= max_lines:
                    break
                yield line.strip()
                count += 1

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()


# Global instance
gcs_loader = GCSDataLoader()
