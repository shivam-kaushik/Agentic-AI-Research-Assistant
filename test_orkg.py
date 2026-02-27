import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tools.orkg_loader import orkg_loader
from tools.search_utils import gemini_filter

query = "Idiopathic Pulmonary Fibrosis"
matches = orkg_loader.multi_search(
    disease_variants=[query],
    topic_keywords=[],
    gene_variants=[]
)

print(f"Raw matches: {len(matches)}")
print("\nFirst 10 matches objects:")
for i, row in matches.head(10).iterrows():
    print(f"Index {i} object:")
    print(repr(row['object']))

if not matches.empty:
    filtered = gemini_filter(matches, "object", query, max_results=30)
    print(f"Filtered matches: {len(filtered)}")
    print(filtered.head())
