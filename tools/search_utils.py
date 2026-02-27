"""
Search Utilities for Co-Investigator Agent.
Provides fuzzy matching and Gemini-based filtering for search results.
"""
import json
from typing import Optional

import pandas as pd
from thefuzz import fuzz
import vertexai
from vertexai.generative_models import GenerativeModel

from config.gcp_config import config


# Initialize Vertex AI
vertexai.init(project=config.project_id, location=config.location)


def smart_search(
    df: pd.DataFrame,
    column: str,
    terms: list[str],
    threshold: int = 90,
    min_len: int = 8,
) -> pd.DataFrame:
    """
    Smart search with fuzzy matching using thefuzz library.

    Args:
        df: DataFrame to search in
        column: Column name to search
        terms: List of search terms
        threshold: Minimum fuzzy match score (0-100)
        min_len: Minimum term length to consider

    Returns:
        DataFrame with matching rows
    """
    if df.empty or not terms:
        return pd.DataFrame(columns=df.columns)

    matched_indices = set()

    for term in terms:
        if not isinstance(term, str) or len(term) < min_len:
            continue

        term_lower = term.lower()

        for idx, val in df[column].items():
            if pd.isna(val):
                continue

            val_str = str(val).lower()

            # Exact substring match
            if term_lower in val_str:
                matched_indices.add(idx)
                continue

            # Fuzzy partial match
            if fuzz.partial_ratio(term_lower, val_str) >= threshold:
                matched_indices.add(idx)

    if not matched_indices:
        return pd.DataFrame(columns=df.columns)

    return df.loc[list(matched_indices)].copy()


def gemini_filter(
    df: pd.DataFrame,
    column: str,
    topic: str,
    max_results: int = 10,
    model_name: str = "gemini-2.5-flash",
) -> pd.DataFrame:
    """
    Use Gemini to filter search results for relevance.

    Args:
        df: DataFrame with search results
        column: Column containing text to evaluate
        topic: Topic/disease to filter for relevance
        max_results: Maximum number of results to return
        model_name: Gemini model to use

    Returns:
        DataFrame with filtered results
    """
    if df.empty:
        return df

    # Take a sample for Gemini to evaluate
    sample_texts = []
    for i, val in enumerate(df[column].head(40)):
        sample_texts.append(f"Index {i}: {str(val)}")
    sample_text = "\n".join(sample_texts)

    prompt = f"""
    You are an expert biomedical researcher filtering knowledge graph text.
    Topic: "{topic}"

    Below is a list of candidate text segments.
    Your task is to identify which segments are relevant to the topic "{topic}".
    Even a simple mention of the topic or its abbreviations is considered relevant.

    {sample_text}

    Return ONLY a JSON array of the indices of relevant segments.
    Example: [0, 1, 4]
    If none are relevant, return [].
    Do not include markdown blocks like ```json.
    """

    try:
        model = GenerativeModel(model_name)
        response = model.generate_content(prompt)
        text = response.text.strip()

        # Clean up response
        text = text.replace("```json", "").replace("```", "").strip()
        print(f"   [gemini_filter DEBUG] Raw Response: {text}")
        indices = json.loads(text)

        # Validate indices
        valid_indices = [
            i for i in indices
            if isinstance(i, int) and 0 <= i < len(df)
        ]

        if not valid_indices:
            return pd.DataFrame(columns=df.columns)

        return df.iloc[valid_indices[:max_results]].copy()

    except Exception as e:
        print(f"   ⚠️  Gemini filter failed: {e}")
        # Fallback: return top N results
        return df.head(max_results).copy()


def safe_len(obj) -> int:
    """Safely get the length of an object."""
    if obj is None:
        return 0
    if isinstance(obj, pd.DataFrame):
        return len(obj)
    try:
        return len(obj)
    except Exception:
        return 0


def combine_search_results(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Combine multiple DataFrames and remove duplicates."""
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        return pd.DataFrame()

    combined = pd.concat(valid_dfs, ignore_index=True)
    return combined.drop_duplicates().reset_index(drop=True)


def search_all_datasets(
    disease_variants: list[str],
    topic_keywords: list[str],
    gene_variants: list[str],
    primary_term: str,
) -> dict:
    """
    Search all datasets with the given terms.

    Returns a dictionary with results from each dataset:
    - clingen: Gene-disease associations
    - pubmedqa: Q&A pairs
    - biorxiv: Preprints
    - orkg: Knowledge graph triples
    """
    from tools.clingen_loader import clingen_loader
    from tools.pubmedqa_loader import pubmedqa_loader
    from tools.biorxiv_loader import biorxiv_loader
    from tools.orkg_loader import orkg_loader

    results = {}

    # Search ClinGen
    print("\n🧬 Searching ClinGen...")
    df_clingen = clingen_loader.load_all()
    clingen_results = smart_search(
        df_clingen, "Disease_Label", disease_variants, threshold=85
    )
    if gene_variants:
        gene_hits = smart_search(
            df_clingen, "Gene_Symbol", gene_variants, threshold=95
        )
        clingen_results = combine_search_results(clingen_results, gene_hits)

    if not clingen_results.empty and primary_term:
        clingen_results = gemini_filter(
            clingen_results, "Disease_Label", primary_term, max_results=15
        )

    results["clingen"] = clingen_results
    print(f"   ✅ ClinGen: {len(clingen_results)} results")

    # Search PubMedQA
    print("\n❓ Searching PubMedQA...")
    df_pubmedqa = pubmedqa_loader.load_searchable()
    all_terms = disease_variants + topic_keywords + gene_variants

    raw_q = smart_search(df_pubmedqa, "Question", all_terms, threshold=80)
    raw_c = smart_search(df_pubmedqa, "Context", all_terms, threshold=80)
    pubmedqa_raw = combine_search_results(raw_q, raw_c)

    if not pubmedqa_raw.empty:
        pubmedqa_results = gemini_filter(
            pubmedqa_raw, "Question", primary_term, max_results=10
        )
    else:
        pubmedqa_results = pd.DataFrame(columns=df_pubmedqa.columns)

    results["pubmedqa"] = pubmedqa_results
    print(f"   ✅ PubMedQA: {len(pubmedqa_results)} results")

    # Search bioRxiv/medRxiv
    print("\n📰 Searching bioRxiv/medRxiv...")
    df_biorxiv = biorxiv_loader.load_all()

    raw_t = smart_search(df_biorxiv, "Title", all_terms, threshold=85)
    raw_a = smart_search(df_biorxiv, "Abstract", all_terms, threshold=85)
    biorxiv_raw = combine_search_results(raw_t, raw_a)

    if not biorxiv_raw.empty:
        biorxiv_results = gemini_filter(
            biorxiv_raw, "Title", primary_term, max_results=10
        )
    else:
        biorxiv_results = pd.DataFrame(columns=df_biorxiv.columns)

    results["biorxiv"] = biorxiv_results
    print(f"   ✅ bioRxiv/medRxiv: {len(biorxiv_results)} results")

    # Search ORKG
    print("\n🔬 Searching ORKG...")
    orkg_results = orkg_loader.multi_search(
        disease_variants=disease_variants,
        topic_keywords=topic_keywords,
        gene_variants=gene_variants,
    )
    results["orkg"] = orkg_results
    print(f"   ✅ ORKG: {len(orkg_results)} results")

    return results


def summarize_search_results(results: dict) -> str:
    """Generate a text summary of search results."""
    lines = []

    for dataset, df in results.items():
        count = safe_len(df)
        lines.append(f"- {dataset.upper()}: {count} results")

        if dataset == "clingen" and count > 0:
            definitive = len(df[df["Classification"] == "Definitive"])
            strong = len(df[df["Classification"] == "Strong"])
            lines.append(f"  └─ {definitive} definitive, {strong} strong")

        elif dataset == "biorxiv" and count > 0:
            bx = len(df[df["source"] == "biorxiv"])
            mx = len(df[df["source"] == "medrxiv"])
            lines.append(f"  └─ {bx} bioRxiv, {mx} medRxiv")

        elif dataset == "pubmedqa" and count > 0:
            yes = len(df[df["Answer"] == "YES"])
            no = len(df[df["Answer"] == "NO"])
            lines.append(f"  └─ {yes} YES, {no} NO answers")

    return "\n".join(lines)
