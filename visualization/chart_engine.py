"""
Chart Engine for Co-Investigator Agent

Generates visualizations with Gemini-powered scientific explanations.
Based on QueryQuest v9.0 Layer 6 visualization engine.
"""
import re
from collections import Counter
from io import BytesIO
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
import vertexai
from vertexai.generative_models import GenerativeModel

from config.gcp_config import config

# Try to import wordcloud (optional)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Dark theme for plots (matching QueryQuest)
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "font.family": "DejaVu Sans",
})

PALETTE = [
    "#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff",
    "#79c0ff", "#56d364", "#e3b341", "#ff7b72", "#d2a8ff"
]


class ChartEngine:
    """Generate charts with Gemini-powered scientific explanations."""

    def __init__(self, model_name: str = "gemini-2.5-pro"):
        vertexai.init(project=config.project_id, location=config.location)
        self.model = GenerativeModel(model_name)

    def gemini_explain_chart(
        self,
        chart_type: str,
        data_summary: str,
        disease_name: str
    ) -> str:
        """
        Generate a 3-sentence scientific explanation for a chart.

        Returns exactly 3 sentences:
        1. Pattern the chart shows (use actual numbers)
        2. Most important scientific insight
        3. One concrete next action for the scientist
        """
        prompt = f"""
        You are a biomedical research analyst explaining a data visualization
        to a preclinical scientist investigating {disease_name}.

        Chart: {chart_type}
        Data: {data_summary}

        Write exactly 3 sentences:
        1. Describe the pattern the chart shows (use actual numbers).
        2. State the most important scientific insight from this pattern.
        3. Suggest one concrete next action for the scientist.

        Max 80 words. Be specific. Use the actual data values.
        Never say "further research is needed".
        """

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Chart generated successfully. ({str(e)[:60]})"

    def viz_gene_classifications(
        self,
        clingen_df: pd.DataFrame,
        disease_name: str
    ) -> Tuple[plt.Figure, str]:
        """
        Horizontal bar chart of ClinGen gene classifications.

        Returns: (figure, explanation)
        """
        if clingen_df is None or clingen_df.empty:
            return None, "No gene data to visualize."

        # Classification order and colors
        order = [
            "Definitive", "Strong", "Moderate", "Limited",
            "Disputed", "No Known Disease Relationship"
        ]
        colors_map = {
            "Definitive": "#3fb950",
            "Strong": "#58a6ff",
            "Moderate": "#d29922",
            "Limited": "#e3b341",
            "Disputed": "#f85149",
            "No Known Disease Relationship": "#8b949e",
        }

        counts = clingen_df["Classification"].value_counts()
        labels = [l for l in order if l in counts.index]
        values = [counts[l] for l in labels]
        colors = [colors_map.get(l, "#8b949e") for l in labels]

        fig, ax = plt.subplots(figsize=(9, 4))
        bars = ax.barh(labels, values, color=colors, edgecolor="#30363d", height=0.55)

        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + 0.05,
                bar.get_y() + bar.get_height() / 2,
                str(val),
                va="center",
                fontsize=11,
                color="#e6edf3",
                fontweight="bold"
            )

        ax.set_xlabel("Number of Gene-Disease Links", fontsize=11)
        ax.set_title(
            f"Gene Classification Breakdown - {disease_name}",
            fontsize=13,
            fontweight="bold",
            color="#58a6ff",
            pad=12
        )
        ax.invert_yaxis()
        ax.grid(axis="x")
        ax.set_xlim(0, max(values) * 1.2 + 1 if values else 1)
        plt.tight_layout()

        # Generate explanation
        top_label = labels[0] if labels else "none"
        top_val = values[0] if values else 0
        summary = (
            f"{len(clingen_df)} gene-disease links for {disease_name}. "
            f"Largest: '{top_label}' with {top_val} entries. "
            f"Breakdown: {dict(zip(labels, values))}"
        )
        explanation = self.gemini_explain_chart(
            "Horizontal bar chart of gene-disease classification levels",
            summary,
            disease_name
        )

        return fig, explanation

    def viz_gene_network(
        self,
        clingen_df: pd.DataFrame,
        disease_name: str
    ) -> Tuple[plt.Figure, str]:
        """
        Network graph of gene-disease associations.

        Returns: (figure, explanation)
        """
        if clingen_df is None or clingen_df.empty:
            return None, "No gene data for network."

        df_plot = clingen_df.head(15).copy()
        G = nx.Graph()

        clf_colors = {
            "Definitive": "#3fb950",
            "Strong": "#58a6ff",
            "Moderate": "#d29922",
            "Disputed": "#f85149",
        }
        node_colors = {disease_name: "#bc8cff"}

        G.add_node(disease_name, node_type="disease")

        for _, row in df_plot.iterrows():
            gene = str(row.get("Gene_Symbol", ""))
            clf = str(row.get("Classification", ""))
            if gene and gene not in ("N/A", "nan", ""):
                G.add_node(gene, node_type="gene")
                G.add_edge(disease_name, gene, classification=clf)
                node_colors[gene] = clf_colors.get(clf, "#8b949e")

        if len(G.nodes()) <= 1:
            return None, "Not enough nodes for network."

        fig, ax = plt.subplots(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2.5, seed=42)
        nodes_list = list(G.nodes())
        colors_list = [node_colors.get(n, "#8b949e") for n in nodes_list]
        sizes = [
            1800 if G.nodes[n].get("node_type") == "disease" else 900
            for n in nodes_list
        ]

        nx.draw_networkx_nodes(
            G, pos, node_color=colors_list, node_size=sizes, alpha=0.9, ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, font_size=8, font_color="#e6edf3", font_weight="bold", ax=ax
        )
        nx.draw_networkx_edges(
            G, pos, edge_color="#30363d", width=1.5, alpha=0.7, ax=ax
        )

        ax.legend(
            handles=[
                mpatches.Patch(color="#bc8cff", label="Disease"),
                mpatches.Patch(color="#3fb950", label="Definitive"),
                mpatches.Patch(color="#58a6ff", label="Strong"),
                mpatches.Patch(color="#d29922", label="Moderate"),
                mpatches.Patch(color="#f85149", label="Disputed"),
            ],
            loc="upper left",
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#e6edf3",
            fontsize=9
        )
        ax.set_title(
            f"Gene-Disease Network - {disease_name}",
            fontsize=13,
            fontweight="bold",
            color="#58a6ff",
            pad=12
        )
        ax.axis("off")
        plt.tight_layout()

        # Generate explanation
        def_genes = clingen_df[
            clingen_df["Classification"] == "Definitive"
        ]["Gene_Symbol"].tolist()
        explanation = self.gemini_explain_chart(
            "Network graph of gene-disease associations colour-coded by classification",
            f"Network: {disease_name} -> {len(df_plot)} genes. Definitive: {def_genes[:5] or 'none'}.",
            disease_name
        )

        return fig, explanation

    def viz_preprints_timeline(
        self,
        biorxiv_df: pd.DataFrame,
        disease_name: str
    ) -> Tuple[plt.Figure, str]:
        """
        Line chart of preprints published over time.

        Returns: (figure, explanation)
        """
        if biorxiv_df is None or biorxiv_df.empty:
            return None, "No preprint data."

        df_plot = biorxiv_df.copy()
        df_plot["Date"] = pd.to_datetime(df_plot["Date"], errors="coerce")
        df_plot = df_plot.dropna(subset=["Date"])

        if df_plot.empty:
            return None, "No valid dates."

        df_plot["YearMonth"] = df_plot["Date"].dt.to_period("M").astype(str)
        counts = df_plot.groupby(["YearMonth", "source"]).size().reset_index(name="Count")
        periods = sorted(counts["YearMonth"].unique())
        x_map = {p: i for i, p in enumerate(periods)}

        fig, ax = plt.subplots(figsize=(10, 4))

        for i, src in enumerate(["biorxiv", "medrxiv"]):
            sub = counts[counts["source"] == src].copy()
            if sub.empty:
                continue
            xs = [x_map[p] for p in sub["YearMonth"] if p in x_map]
            ys = sub["Count"].tolist()
            ax.plot(
                xs, ys, marker="o", linewidth=2.5, markersize=7,
                color=PALETTE[i], label=src.capitalize()
            )
            ax.fill_between(xs, ys, alpha=0.08, color=PALETTE[i])

        ax.set_xticks(list(range(len(periods))))
        ax.set_xticklabels(periods, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Number of Preprints", fontsize=11)
        ax.set_title(
            f"Preprint Publication Trend - {disease_name}",
            fontsize=13,
            fontweight="bold",
            color="#58a6ff",
            pad=12
        )
        ax.legend(
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#e6edf3"
        )
        ax.grid(axis="y")
        plt.tight_layout()

        # Generate explanation
        bx_n = len(biorxiv_df[biorxiv_df["source"] == "biorxiv"])
        mx_n = len(biorxiv_df[biorxiv_df["source"] == "medrxiv"])
        dr = (
            f"{df_plot['Date'].min().strftime('%b %Y')} "
            f"to {df_plot['Date'].max().strftime('%b %Y')}"
        )
        explanation = self.gemini_explain_chart(
            "Line chart of preprints published per month",
            f"{len(biorxiv_df)} preprints, {dr}. bioRxiv:{bx_n} medRxiv:{mx_n}.",
            disease_name
        )

        return fig, explanation

    def viz_keyword_wordcloud(
        self,
        biorxiv_df: pd.DataFrame,
        disease_name: str
    ) -> Tuple[plt.Figure, str]:
        """
        Word cloud from abstract keywords.

        Returns: (figure, explanation)
        """
        if not WORDCLOUD_AVAILABLE:
            return None, "WordCloud not installed."

        if biorxiv_df is None or biorxiv_df.empty:
            return None, "No abstracts."

        STOPWORDS = {
            "the", "and", "or", "in", "of", "to", "a", "an", "is", "are", "was", "were",
            "that", "this", "for", "with", "on", "at", "by", "from", "as", "we", "our",
            "these", "their", "which", "also", "can", "may", "been", "have", "has",
            "study", "studies", "data", "results", "patients", "disease", "diseases",
            "using", "used", "methods", "analysis", "model", "models", "showed", "show",
            "found", "finding", "findings", "suggest", "suggests", "including",
            "associated", "association", "compared", "between", "within", "among",
            "however", "although", "therefore", "furthermore", "moreover", "while",
        }

        all_text = " ".join(biorxiv_df["Abstract"].fillna("").tolist())
        words = [
            w.lower() for w in re.findall(r"[a-zA-Z]{5,}", all_text)
            if w.lower() not in STOPWORDS
        ]

        if not words:
            return None, "No words extracted."

        top_words = Counter(words).most_common(20)

        wc = WordCloud(
            width=1000,
            height=450,
            max_words=80,
            background_color="#0d1117",
            colormap="cool",
            contour_color="#30363d",
            contour_width=1,
            collocations=False
        ).generate(" ".join(words))

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(
            f"Top Research Keywords - {disease_name}",
            fontsize=13,
            fontweight="bold",
            color="#58a6ff",
            pad=12
        )
        plt.tight_layout()

        # Generate explanation
        top5 = ", ".join([f"'{w}' ({c}x)" for w, c in top_words[:5]])
        explanation = self.gemini_explain_chart(
            "Word cloud of most frequent abstract terms",
            f"{len(biorxiv_df)} abstracts. Top 5: {top5}.",
            disease_name
        )

        return fig, explanation

    def viz_researcher_ranking(
        self,
        researchers_df: pd.DataFrame,
        disease_name: str
    ) -> Tuple[plt.Figure, str]:
        """
        Side-by-side bar chart for H-index and citations.

        Returns: (figure, explanation)
        """
        if researchers_df is None or researchers_df.empty:
            return None, "No researcher data."

        df_plot = researchers_df.head(10).copy().sort_values("h_index", ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(df_plot) * 0.7)))

        # H-Index chart
        axes[0].barh(
            df_plot["name"],
            df_plot["h_index"].astype(float),
            color=PALETTE[0],
            edgecolor="#30363d",
            height=0.6
        )
        for i, (_, row) in enumerate(df_plot.iterrows()):
            axes[0].text(
                float(row["h_index"]) + 0.3,
                i,
                str(row["h_index"]),
                va="center",
                fontsize=9,
                color="#e6edf3",
                fontweight="bold"
            )
        axes[0].set_xlabel("H-Index", fontsize=11)
        axes[0].set_title("H-Index", fontsize=12, fontweight="bold", color="#58a6ff")
        axes[0].grid(axis="x")
        axes[0].set_xlim(0, df_plot["h_index"].astype(float).max() * 1.25 + 1)

        # Citations chart
        axes[1].barh(
            df_plot["name"],
            df_plot["cited_by_count"].astype(float),
            color=PALETTE[1],
            edgecolor="#30363d",
            height=0.6
        )
        for i, (_, row) in enumerate(df_plot.iterrows()):
            val = int(row["cited_by_count"])
            axes[1].text(
                val + 100,
                i,
                f"{val:,}",
                va="center",
                fontsize=9,
                color="#e6edf3",
                fontweight="bold"
            )
        axes[1].set_xlabel("Total Citations", fontsize=11)
        axes[1].set_title(
            "Total Citations",
            fontsize=12,
            fontweight="bold",
            color="#3fb950"
        )
        axes[1].grid(axis="x")
        axes[1].set_xlim(0, df_plot["cited_by_count"].astype(float).max() * 1.25 + 1)

        fig.suptitle(
            f"Top Researchers - {disease_name}",
            fontsize=14,
            fontweight="bold",
            color="#e6edf3",
            y=1.02
        )
        plt.tight_layout()

        # Generate explanation
        top_r = researchers_df.iloc[0]
        names = ", ".join(researchers_df["name"].head(5).tolist())
        explanation = self.gemini_explain_chart(
            "Side-by-side bar chart: H-index and citations per researcher",
            (
                f"Top {len(researchers_df)} researchers. Best: {top_r['name']} "
                f"(H:{top_r['h_index']}, {int(top_r['cited_by_count']):,} citations). "
                f"All: {names}."
            ),
            disease_name
        )

        return fig, explanation

    def viz_research_network(
        self,
        researchers_df: pd.DataFrame,
        disease_name: str,
        clingen_df: Optional[pd.DataFrame] = None
    ) -> Tuple[plt.Figure, str]:
        """
        Network of researchers, disease, and genes.

        Returns: (figure, explanation)
        """
        if researchers_df is None or researchers_df.empty:
            return None, "No researcher data for network."

        G = nx.Graph()
        node_colors = {}

        G.add_node(disease_name, node_type="disease")
        node_colors[disease_name] = "#bc8cff"

        # Add researchers
        for _, row in researchers_df.head(8).iterrows():
            name = row["name"]
            G.add_node(name, node_type="researcher")
            G.add_edge(name, disease_name, weight=int(row.get("h_index", 1)))
            node_colors[name] = "#58a6ff"

        # Add definitive genes if available
        if clingen_df is not None and not clingen_df.empty:
            definitive = clingen_df[clingen_df["Classification"] == "Definitive"]
            for _, row in definitive.head(5).iterrows():
                gene = row["Gene_Symbol"]
                G.add_node(gene, node_type="gene")
                G.add_edge(disease_name, gene)
                node_colors[gene] = "#3fb950"

        fig, ax = plt.subplots(figsize=(13, 9))
        pos = nx.spring_layout(G, k=3.0, seed=42)
        nodes_list = list(G.nodes())
        colors_list = [node_colors.get(n, "#8b949e") for n in nodes_list]
        sizes = [
            2200 if G.nodes[n].get("node_type") == "disease"
            else 1200 if G.nodes[n].get("node_type") == "researcher"
            else 900
            for n in nodes_list
        ]

        nx.draw_networkx_nodes(
            G, pos, node_color=colors_list, node_size=sizes, alpha=0.9, ax=ax
        )
        nx.draw_networkx_labels(
            G, pos, font_size=8, font_color="#e6edf3", font_weight="bold", ax=ax
        )
        nx.draw_networkx_edges(
            G, pos, edge_color="#30363d", width=1.8, alpha=0.6, ax=ax
        )

        ax.legend(
            handles=[
                mpatches.Patch(color="#bc8cff", label="Disease"),
                mpatches.Patch(color="#58a6ff", label="Researcher"),
                mpatches.Patch(color="#3fb950", label="Gene (Definitive)"),
            ],
            loc="upper left",
            facecolor="#161b22",
            edgecolor="#30363d",
            labelcolor="#e6edf3",
            fontsize=9
        )
        ax.set_title(
            f"Research Network - {disease_name}",
            fontsize=13,
            fontweight="bold",
            color="#58a6ff",
            pad=12
        )
        ax.axis("off")
        plt.tight_layout()

        # Generate explanation
        gc = len(clingen_df[clingen_df["Classification"] == "Definitive"]) if clingen_df is not None else 0
        res = ", ".join(researchers_df["name"].head(5).tolist())
        explanation = self.gemini_explain_chart(
            "Network: disease node connected to researchers and genes",
            f"Disease: {disease_name}. Researchers: {len(researchers_df)}. "
            f"Definitive genes: {gc}. Names: {res}.",
            disease_name
        )

        return fig, explanation

    def fig_to_bytes(self, fig: plt.Figure) -> bytes:
        """Convert a figure to PNG bytes for display."""
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        return buf.getvalue()


def generate_all_visualizations(state: dict, disease_name: str) -> list[dict]:
    """
    Generate all available visualizations from state.

    Returns list of dicts: {name, figure, explanation}
    """
    engine = ChartEngine()
    charts = []

    # ClinGen results
    clingen_results = state.get("clingen_results")
    if clingen_results and isinstance(clingen_results, dict):
        all_results = clingen_results.get("all_results", [])
        if all_results:
            df = pd.DataFrame(all_results)

            fig, expl = engine.viz_gene_classifications(df, disease_name)
            if fig:
                charts.append({
                    "name": "Gene Classifications",
                    "figure": fig,
                    "explanation": expl,
                })

            fig, expl = engine.viz_gene_network(df, disease_name)
            if fig:
                charts.append({
                    "name": "Gene-Disease Network",
                    "figure": fig,
                    "explanation": expl,
                })

    # bioRxiv results
    biorxiv_results = state.get("biorxiv_results")
    if biorxiv_results and isinstance(biorxiv_results, dict):
        results = biorxiv_results.get("results", [])
        if results:
            df = pd.DataFrame(results)

            fig, expl = engine.viz_preprints_timeline(df, disease_name)
            if fig:
                charts.append({
                    "name": "Preprint Timeline",
                    "figure": fig,
                    "explanation": expl,
                })

            fig, expl = engine.viz_keyword_wordcloud(df, disease_name)
            if fig:
                charts.append({
                    "name": "Keyword Cloud",
                    "figure": fig,
                    "explanation": expl,
                })

    # Researcher results
    researcher_results = state.get("researcher_results")
    if researcher_results:
        researchers = researcher_results
        if isinstance(researcher_results, dict):
            researchers = researcher_results.get("researchers", [])
        if researchers:
            df = pd.DataFrame(researchers)

            fig, expl = engine.viz_researcher_ranking(df, disease_name)
            if fig:
                charts.append({
                    "name": "Researcher Ranking",
                    "figure": fig,
                    "explanation": expl,
                })

            # Also include ClinGen for research network
            clingen_df = None
            if clingen_results and isinstance(clingen_results, dict):
                all_results = clingen_results.get("all_results", [])
                if all_results:
                    clingen_df = pd.DataFrame(all_results)

            fig, expl = engine.viz_research_network(df, disease_name, clingen_df)
            if fig:
                charts.append({
                    "name": "Research Network",
                    "figure": fig,
                    "explanation": expl,
                })

    return charts
