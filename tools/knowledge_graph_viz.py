"""
Knowledge Graph Visualization for Co-Investigator Agent.
Generates visual network graphs from ORKG triples using networkx and matplotlib.
"""
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    import networkx as nx
    HAS_VIZ_LIBS = True
except ImportError:
    HAS_VIZ_LIBS = False
    logger.warning("networkx or matplotlib not available. Knowledge graph visualization disabled.")


def create_knowledge_graph(
    orkg_data: list[dict],
    disease_name: str,
    gene_symbols: list[str] = None,
    output_dir: str = "outputs",
    session_id: str = None,
) -> dict:
    """
    Create a visual knowledge graph from ORKG triples.

    Args:
        orkg_data: List of ORKG triple dictionaries with 'subject', 'predicate', 'object'
        disease_name: Primary disease name for the central node
        gene_symbols: List of gene symbols to highlight
        output_dir: Directory to save the graph image
        session_id: Session ID for unique filename

    Returns:
        Dictionary with graph info and file path
    """
    if not HAS_VIZ_LIBS:
        return {
            "success": False,
            "error": "Visualization libraries (networkx, matplotlib) not installed",
            "graph_path": None,
            "node_count": 0,
            "edge_count": 0,
        }

    if not orkg_data:
        return {
            "success": False,
            "error": "No ORKG data to visualize",
            "graph_path": None,
            "node_count": 0,
            "edge_count": 0,
        }

    try:
        # Create graph
        G = nx.DiGraph()

        # Add disease as central node
        disease_node = disease_name[:30]
        G.add_node(disease_node, node_type="disease", color="#FF6B6B", size=3000)

        # Add gene nodes if provided
        gene_symbols = gene_symbols or []
        for gene in gene_symbols[:5]:  # Limit to 5 genes
            G.add_node(gene, node_type="gene", color="#4ECDC4", size=2000)
            G.add_edge(disease_node, gene, relationship="associated_gene")

        # Process ORKG triples
        concept_nodes = set()
        for triple in orkg_data[:20]:  # Limit to 20 triples
            subj = str(triple.get("subject", ""))
            obj = str(triple.get("object", ""))
            pred = str(triple.get("predicate", "related_to"))

            # Clean subject (extract ID from URI)
            if "/" in subj:
                subj = subj.split("/")[-1]
            subj = subj[:25]

            # Clean object
            obj = obj[:40]

            if not obj or len(obj) < 5:
                continue

            # Add concept node
            if obj not in concept_nodes:
                G.add_node(obj, node_type="concept", color="#95E1D3", size=1500)
                concept_nodes.add(obj)

            # Connect to disease or a gene if relevant
            if any(gene.lower() in obj.lower() for gene in gene_symbols):
                for gene in gene_symbols:
                    if gene.lower() in obj.lower():
                        G.add_edge(gene, obj, relationship="mechanism")
                        break
            else:
                G.add_edge(disease_node, obj, relationship="concept")

        if len(G.nodes()) < 2:
            return {
                "success": False,
                "error": "Not enough nodes to create a meaningful graph",
                "graph_path": None,
                "node_count": len(G.nodes()),
                "edge_count": len(G.edges()),
            }

        # Create figure with larger size
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        # Calculate layout
        if len(G.nodes()) > 10:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)

        # Get node colors and sizes
        node_colors = [G.nodes[n].get('color', '#95E1D3') for n in G.nodes()]
        node_sizes = [G.nodes[n].get('size', 1500) for n in G.nodes()]

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#4a4a6a',
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            width=1.5,
            connectionstyle="arc3,rad=0.1"
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.9,
            edgecolors='white',
            linewidths=2
        )

        # Draw labels
        labels = {n: n[:20] + '...' if len(n) > 20 else n for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos, labels, ax=ax,
            font_size=9,
            font_color='white',
            font_weight='bold'
        )

        # Add title and legend
        ax.set_title(
            f"Knowledge Graph: {disease_name}",
            fontsize=16,
            fontweight='bold',
            color='white',
            pad=20
        )

        # Create legend
        legend_elements = [
            plt.scatter([], [], c='#FF6B6B', s=200, label='Disease', edgecolors='white'),
            plt.scatter([], [], c='#4ECDC4', s=150, label='Gene', edgecolors='white'),
            plt.scatter([], [], c='#95E1D3', s=100, label='Concept', edgecolors='white'),
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper left',
            facecolor='#2a2a4a',
            edgecolor='white',
            labelcolor='white',
            fontsize=10
        )

        ax.axis('off')
        plt.tight_layout()

        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"knowledge_graph_{session_id or 'default'}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)

        logger.info(f"Knowledge graph saved to {filepath}")

        return {
            "success": True,
            "graph_path": filepath,
            "node_count": len(G.nodes()),
            "edge_count": len(G.edges()),
            "disease_node": disease_node,
            "gene_nodes": gene_symbols[:5],
            "concept_nodes": list(concept_nodes)[:15],
        }

    except Exception as e:
        logger.error(f"Failed to create knowledge graph: {e}")
        return {
            "success": False,
            "error": str(e),
            "graph_path": None,
            "node_count": 0,
            "edge_count": 0,
        }


def create_gene_disease_graph(
    clingen_data: list[dict],
    disease_name: str,
    output_dir: str = "outputs",
    session_id: str = None,
) -> dict:
    """
    Create a gene-disease relationship graph from ClinGen data.

    Args:
        clingen_data: List of ClinGen records
        disease_name: Primary disease name
        output_dir: Directory to save the graph
        session_id: Session ID for unique filename

    Returns:
        Dictionary with graph info and file path
    """
    if not HAS_VIZ_LIBS:
        return {
            "success": False,
            "error": "Visualization libraries not installed",
            "graph_path": None,
        }

    if not clingen_data:
        return {
            "success": False,
            "error": "No ClinGen data to visualize",
            "graph_path": None,
        }

    try:
        G = nx.Graph()

        # Add disease as central node
        disease_node = disease_name[:25]
        G.add_node(disease_node, node_type="disease", color="#FF6B6B", size=4000)

        # Classification colors
        class_colors = {
            "Definitive": "#2ECC71",  # Green
            "Strong": "#3498DB",       # Blue
            "Moderate": "#F39C12",     # Orange
            "Limited": "#E74C3C",      # Red
            "Disputed": "#9B59B6",     # Purple
        }

        # Add gene nodes
        for record in clingen_data[:15]:  # Limit to 15 genes
            gene = record.get("Gene_Symbol", "")
            classification = record.get("Classification", "Limited")
            moi = record.get("MOI", "")

            if not gene:
                continue

            color = class_colors.get(classification, "#95A5A6")
            size = {
                "Definitive": 2500,
                "Strong": 2200,
                "Moderate": 1900,
                "Limited": 1600,
            }.get(classification, 1500)

            G.add_node(gene, node_type="gene", color=color, size=size,
                      classification=classification, moi=moi)
            G.add_edge(disease_node, gene, classification=classification)

        if len(G.nodes()) < 2:
            return {"success": False, "error": "Not enough data", "graph_path": None}

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')

        # Layout
        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)

        # Get colors and sizes
        node_colors = [G.nodes[n].get('color', '#95A5A6') for n in G.nodes()]
        node_sizes = [G.nodes[n].get('size', 1500) for n in G.nodes()]

        # Draw
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#4a4a6a', alpha=0.6, width=2)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=node_sizes, alpha=0.9,
                               edgecolors='white', linewidths=2)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10,
                                font_color='white', font_weight='bold')

        ax.set_title(f"Gene-Disease Relationships: {disease_name}",
                    fontsize=16, fontweight='bold', color='white', pad=20)

        # Legend
        legend_elements = [
            plt.scatter([], [], c='#FF6B6B', s=200, label='Disease', edgecolors='white'),
            plt.scatter([], [], c='#2ECC71', s=150, label='Definitive', edgecolors='white'),
            plt.scatter([], [], c='#3498DB', s=130, label='Strong', edgecolors='white'),
            plt.scatter([], [], c='#F39C12', s=110, label='Moderate', edgecolors='white'),
            plt.scatter([], [], c='#E74C3C', s=90, label='Limited', edgecolors='white'),
        ]
        ax.legend(handles=legend_elements, loc='upper left',
                 facecolor='#2a2a4a', edgecolor='white',
                 labelcolor='white', fontsize=10)

        ax.axis('off')
        plt.tight_layout()

        # Save
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gene_disease_graph_{session_id or 'default'}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
        plt.close(fig)

        return {
            "success": True,
            "graph_path": filepath,
            "node_count": len(G.nodes()),
            "edge_count": len(G.edges()),
        }

    except Exception as e:
        logger.error(f"Failed to create gene-disease graph: {e}")
        return {"success": False, "error": str(e), "graph_path": None}
