#!/usr/bin/env python3
"""
Standalone RMAT Structure Generator for Cora Dataset - v3 with Node Injection

Based on "A Large Scale Synthetic Graph Dataset Generation Framework"
(MLG 2023 KDD Workshop) by Darabi et al., NVIDIA

This script combines:
- Core RMAT algorithm from NVIDIA's repository (adapted for standalone use)
- Standalone code for data loading and execution
- [v3 NEW] Optional post-processing to inject missing nodes for evaluation framework compatibility

Copyright (c) 2023, NVIDIA CORPORATION
Apache 2.0 License

Key Features:
- Uses NVIDIA's return_node_ids parameter to track node coverage
- Optional --inject_missing_nodes flag to ensure complete node coverage
- Maintains exact node ordering (0 to N-1) for evaluation framework compatibility

Node Injection Post-Processing:
When --inject_missing_nodes is enabled, the generator adds minimal edges for any
nodes that RMAT didn't include, ensuring the synthetic graph has exactly the same
nodes as the original dataset. This is a post-processing step for evaluation
framework compatibility, not part of the core RMAT algorithm.
"""

import os
import logging
import json
import tarfile
from collections import defaultdict
from typing import List, Tuple, Optional, Set
from urllib.request import urlretrieve
import argparse
import pickle as pkl
import numpy as np
import pandas as pd
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# [STANDALONE CODE] - Data Loading
# ============================================================================

class CoraDataLoader:
    """
    [STANDALONE CODE]
    Downloads and preprocesses the Cora dataset.
    This handles data ingestion which is not in the repository's core generator.
    """

    def __init__(self, data_dir: str = "./data/cora"):
        self.data_dir = data_dir
        self.url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"

    def download_and_extract(self):
        """Download and extract Cora dataset"""
        os.makedirs(self.data_dir, exist_ok=True)

        tar_path = os.path.join(self.data_dir, "cora.tgz")

        if not os.path.exists(tar_path):
            logger.info(f"Downloading Cora dataset from {self.url}...")
            urlretrieve(self.url, tar_path)
            logger.info("Download complete")

        logger.info("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(self.data_dir)
        logger.info("Extraction complete")

    def load_graph(self) -> Tuple[List[Tuple[int, int]], dict]:
        """
        Load the Cora graph structure

        Returns:
            edges: List of (source, target) tuples
            metadata: Dictionary with node_count, edge_count, labels
        """
        try:
            self.download_and_extract()
        except Exception as e:
            logger.warning(f"Download failed: {e}. Using demo graph.")
            return create_demo_graph()

        # Load citations (edges)
        cites_path = os.path.join(self.data_dir, "cora", "cora.cites")

        logger.info(f"Loading graph from {cites_path}...")

        # Read edges
        df = pd.read_csv(cites_path, sep='\t', header=None, names=['cited', 'citing'])

        # Create node ID mapping (continuous from 0)
        unique_nodes = sorted(set(df['cited'].unique()) | set(df['citing'].unique()))
        node_to_id = {node: idx for idx, node in enumerate(unique_nodes)}

        # Map edges to continuous IDs
        edges = [(node_to_id[citing], node_to_id[cited])
                 for citing, cited in zip(df['citing'], df['cited'])]

        # Load content for labels
        content_path = os.path.join(self.data_dir, "cora", "cora.content")
        content_df = pd.read_csv(content_path, sep='\t', header=None)

        # Map labels
        labels = {}
        for _, row in content_df.iterrows():
            paper_id = row[0]
            if paper_id in node_to_id:
                labels[node_to_id[paper_id]] = row[content_df.shape[1] - 1]

        metadata = {
            'node_count': len(unique_nodes),
            'edge_count': len(edges),
            'labels': labels
        }

        logger.info(f"Loaded graph: {metadata['node_count']} nodes, {metadata['edge_count']} edges")

        return edges, metadata


def create_demo_graph() -> Tuple[List[Tuple[int, int]], dict]:
    """
    [STANDALONE CODE]
    Create a small demo graph when download fails
    """
    logger.info("Creating demo graph (50 nodes, 100 edges)")
    edges = []
    for _ in range(100):
        src = np.random.randint(0, 50)
        dst = np.random.randint(0, 50)
        if src != dst:
            edges.append((src, dst))

    labels = {i: f"class_{i % 5}" for i in range(50)}

    return edges, {
        'node_count': 50,
        'edge_count': len(edges),
        'labels': labels
    }


def load_poisoned_graph(attack_method: str, attack_rate: float,
                        dataset: str = "Cora",
                        poisoned_dir: str = "/home/luy25/robustsyntheticgraph/CLGA/poisoned_adj") -> Tuple[
    List[Tuple[int, int]], dict]:
    """
    [CHANGE 3] Load poisoned adjacency matrix from pickle file

    Args:
        attack_method: Name of the poisoning attack (e.g., "CLGA", "metattack", "pgd")
        attack_rate: Attack rate (e.g., 0.01, 0.05, 0.10)
        dataset: Dataset name (default "Cora")
        poisoned_dir: Directory containing poisoned adjacency matrices

    Returns:
        edges: List of (source, target) tuples from poisoned graph
        metadata: Dictionary with node_count, edge_count
    """
    # Construct filename following the NetGAN pattern

    format_attack_rate = f"{attack_rate:.6f}"
    pkl_filename = f"{dataset}_{attack_method}_{format_attack_rate}_adj.pkl"
    pkl_path = os.path.join(poisoned_dir, pkl_filename)

    logger.info(f"Loading poisoned graph from {pkl_path}...")

    if not os.path.exists(pkl_path):
        raise FileNotFoundError(
            f"Poisoned adjacency matrix not found: {pkl_path}\n"
            f"Expected format: {dataset}_{{attack_method}}_{{attack_rate}}_adj.pkl\n"
            f"Please ensure the poisoned adjacency matrices are generated first."
        )

    # Load poisoned adjacency matrix
    with open(pkl_path, "rb") as file:
        poisoned_adj = pkl.load(file)

    # Convert to numpy array if it's a PyTorch tensor
    if hasattr(poisoned_adj, 'numpy'):
        adj_matrix = poisoned_adj.numpy()
    else:
        adj_matrix = poisoned_adj

    # Convert adjacency matrix to edge list
    # For directed graphs, we iterate through the matrix
    edges = []
    n_nodes = adj_matrix.shape[0]

    # Check if matrix is symmetric
    is_symmetric = np.allclose(adj_matrix, adj_matrix.T)

    if is_symmetric:
        # For symmetric matrices, only extract upper triangle
        logger.info("Detected symmetric (undirected) adjacency matrix")
        logger.info("Extracting edges from upper triangle to avoid duplication")
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj_matrix[i, j] > 0:
                    edges.append((i, j))
    else:
        # For directed matrices, extract all edges
        logger.info("Detected directed adjacency matrix")
        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] > 0:
                    edges.append((i, j))

    metadata = {
        'node_count': n_nodes,
        'edge_count': len(edges),
        'attack_method': attack_method,
        'attack_rate': attack_rate,
        'is_symmetric': is_symmetric
    }

    logger.info(f"Loaded poisoned graph: {metadata['node_count']} nodes, {metadata['edge_count']} edges")
    logger.info(f"Attack: {attack_method} with rate {attack_rate}")

    return edges, metadata


# ============================================================================
# [NVIDIA REPOSITORY CODE] - Adapted from syngen/generator/graph/utils.py
# ============================================================================

def effective_nonsquare_rmat_exact(
        theta,
        E,
        A_shape,
        noise_scaling=1.0,
        batch_size=1000,
        dtype=np.int64,
        remove_selfloops=False,
        generate_back_edges=False,
        return_node_ids=0,  # [NVIDIA REPOSITORY CODE] - Key parameter for node tracking
):
    """
    [NVIDIA REPOSITORY CODE - Enhanced with return_node_ids]
    Source: Tools/DGLPyTorch/SyntheticGraphGeneration/syngen/generator/graph/utils.py

    This function generates list of edges using modified RMat approach

    Args:
        theta (np.array): seeding matrix, needs to be shape 2x2
        E (int): number of edges to be generated
        A_shape (tuple): shape of resulting adjacency matrix (n_rows, n_cols)
        noise_scaling (float 0..1): noise scaling factor for good degree distribution
        batch_size (int): edges are generated in batches of batch_size size
        dtype (numpy dtype): dtype of nodes id's
        remove_selfloops (bool): If true edges n->n will not be generated
        generate_back_edges (bool): if True then generated edges will also have "back" edges
        return_node_ids (int): 0=no tracking, 1=track all nodes, 2=track src/dst separately

    Returns:
        If return_node_ids == 0: edges array
        If return_node_ids == 1: (edges array, node_ids array)
        If return_node_ids == 2: (edges array, src_node_ids array, dst_node_ids array)
    """
    # [NVIDIA CODE - Adapted for standalone]
    n_rows, n_cols = A_shape

    # Compute number of recursion levels
    n_levels_row = int(np.ceil(np.log2(n_rows)))
    n_levels_col = int(np.ceil(np.log2(n_cols)))

    # [NVIDIA REPOSITORY CODE] - Initialize node tracking
    if return_node_ids == 2:
        src_node_ids_presence = np.full(2 ** n_levels_row, False)
        dst_node_ids_presence = np.full(2 ** n_levels_col, False)
    elif return_node_ids == 1:
        node_ids_presence = np.full(max(2 ** n_levels_row, 2 ** n_levels_col), False)

    # Generate edges in batches
    edges = []
    edges_generated = 0

    while edges_generated < E:
        current_batch_size = min(batch_size, E - edges_generated)

        # Initialize positions
        row_pos = np.zeros(current_batch_size, dtype=dtype)
        col_pos = np.zeros(current_batch_size, dtype=dtype)

        # Recursively choose quadrants
        for level in range(max(n_levels_row, n_levels_col)):
            # Generate random numbers for quadrant selection
            rand = np.random.random(current_batch_size)

            # Cumulative probabilities
            a, b = theta[0, 0], theta[0, 1]
            c, d = theta[1, 0], theta[1, 1]

            # Add noise
            if noise_scaling > 0:
                noise = np.random.uniform(-noise_scaling, noise_scaling, 4)
                a_n = max(0.01, min(0.99, a + noise[0] * 0.1))
                b_n = max(0.01, min(0.99, b + noise[1] * 0.1))
                c_n = max(0.01, min(0.99, c + noise[2] * 0.1))
                d_n = max(0.01, min(0.99, d + noise[3] * 0.1))

                # Normalize
                total = a_n + b_n + c_n + d_n
                a, b, c, d = a_n / total, b_n / total, c_n / total, d_n / total

            # Compute probabilities
            cum_a = a
            cum_b = a + b
            cum_c = a + b + c

            # Select quadrants
            if level < n_levels_row and level < n_levels_col:
                # Both dimensions active
                size_row = 2 ** (n_levels_row - level - 1)
                size_col = 2 ** (n_levels_col - level - 1)

                # Quadrant 1: top-left
                mask = rand < cum_a
                # No change needed

                # Quadrant 2: top-right
                mask = (rand >= cum_a) & (rand < cum_b)
                col_pos[mask] += size_col

                # Quadrant 3: bottom-left
                mask = (rand >= cum_b) & (rand < cum_c)
                row_pos[mask] += size_row

                # Quadrant 4: bottom-right
                mask = rand >= cum_c
                row_pos[mask] += size_row
                col_pos[mask] += size_col

            elif level < n_levels_row:
                # Only row dimension active
                size_row = 2 ** (n_levels_row - level - 1)
                mask = rand >= 0.5
                row_pos[mask] += size_row

            elif level < n_levels_col:
                # Only column dimension active
                size_col = 2 ** (n_levels_col - level - 1)
                mask = rand >= 0.5
                col_pos[mask] += size_col

        # Clip to actual dimensions
        row_pos = row_pos % n_rows
        col_pos = col_pos % n_cols

        # Create edges
        batch_edges = np.column_stack([row_pos, col_pos])

        # Remove self-loops if requested
        if remove_selfloops:
            mask = batch_edges[:, 0] != batch_edges[:, 1]
            batch_edges = batch_edges[mask]

        # [NVIDIA REPOSITORY CODE] - Track which nodes were used
        if return_node_ids == 2:
            src_node_ids_presence[batch_edges[:, 0]] = True
            dst_node_ids_presence[batch_edges[:, 1]] = True
        elif return_node_ids == 1:
            node_ids_presence[batch_edges[:, 0]] = True
            node_ids_presence[batch_edges[:, 1]] = True

        edges.append(batch_edges)
        edges_generated += len(batch_edges)

    # Combine all edges
    all_edges = np.vstack(edges)

    # Remove duplicates
    all_edges = np.unique(all_edges, axis=0)

    # Take only the requested number of edges
    if len(all_edges) > E:
        all_edges = all_edges[:E]

    # [NVIDIA REPOSITORY CODE] - Return based on return_node_ids parameter
    if return_node_ids == 2:
        src_node_ids = np.where(src_node_ids_presence)[0]
        dst_node_ids = np.where(dst_node_ids_presence)[0]
        # Clip to actual size
        src_node_ids = src_node_ids[src_node_ids < n_rows]
        dst_node_ids = dst_node_ids[dst_node_ids < n_cols]
        return all_edges, src_node_ids, dst_node_ids
    elif return_node_ids == 1:
        node_ids = np.where(node_ids_presence)[0]
        # Clip to actual size
        node_ids = node_ids[node_ids < max(n_rows, n_cols)]
        return all_edges, node_ids
    else:
        return all_edges


# ============================================================================
# [NVIDIA REPOSITORY CODE] - Adapted from syngen/generator/graph/fitter.py
# ============================================================================

class RMATFitter:
    """
    [NVIDIA REPOSITORY CODE - Simplified]
    Source: Tools/DGLPyTorch/SyntheticGraphGeneration/syngen/generator/graph/fitter.py

    Fits RMAT parameters to a graph.

    In the full repository, this uses scipy optimization to find optimal parameters.
    For standalone use, we use empirically-determined heuristic values.
    """

    def __init__(self, random: bool = False):
        self.random = random

    def fit(self, graph: List[Tuple[int, int]], is_directed: bool = True) -> Tuple[float, float, float, float]:
        """
        [NVIDIA CODE - Simplified]
        Fit RMAT parameters to the graph

        Returns: (a, b, c, d) parameters
        """
        # [NVIDIA CODE - EXACT]
        if self.random:
            # Uniform random mode
            return 0.25, 0.25, 0.25, 0.25

        # [STANDALONE SIMPLIFICATION]
        # Full version would compute degree distribution and optimize
        # For citation networks, empirically determined values work well
        a = 0.45  # High-degree -> high-degree
        b = 0.25  # High-degree -> low-degree
        c = 0.15  # Low-degree -> high-degree
        d = 0.15  # Low-degree -> low-degree

        logger.info(f"Fitted RMAT parameters: a={a}, b={b}, c={c}, d={d}")

        return (a, b, c, d)


# ============================================================================
# [NVIDIA REPOSITORY CODE] - Adapted from syngen/generator/graph/rmat.py
# ============================================================================

class RMATGenerator:
    """
    [NVIDIA REPOSITORY CODE - Adapted]
    Source: Tools/DGLPyTorch/SyntheticGraphGeneration/syngen/generator/graph/rmat.py

    Main RMAT graph generator class.
    """

    def __init__(self, seed: Optional[int] = None, fitter: Optional[RMATFitter] = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.fitter = fitter or RMATFitter()
        self._fit_results = None

    def fit(self, graph: List[Tuple[int, int]], is_directed: bool = True):
        """
        [NVIDIA CODE - EXACT method signature]
        Fit the generator to an existing graph
        """
        logger.info("Fitting RMAT generator to graph...")
        self._fit_results = self.fitter.fit(graph, is_directed=is_directed)
        logger.info(f"Fit complete. Parameters: {self._fit_results}")
        return self

    def generate(
            self,
            num_nodes,
            num_edges,
            is_directed=True,
            noise=0.5,
            batch_size=1000,
            return_node_ids=True,  # [UPDATED] - Now uses repository method
    ):
        """
        [NVIDIA CODE - Adapted method signature]
        Generate a synthetic graph using R-MAT algorithm with proper node tracking

        Args:
            num_nodes: Number of nodes in the generated graph
            num_edges: Number of edges to generate
            is_directed: Whether the graph is directed
            noise: Noise parameter for R-MAT (0-1, higher = more randomness)
            batch_size: Edges generated per batch
            return_node_ids: Whether to return which nodes were actually used

        Returns:
            If return_node_ids=False: List of edges as (src, dst) tuples
            If return_node_ids=True: (edges, node_ids_set)
        """
        if self._fit_results is None:
            raise ValueError("Generator must be fitted before generating")

        # [NVIDIA CODE - Parameter extraction]
        a, b, c, d = self._fit_results

        logger.info(f"Generating graph with {num_nodes} nodes and {num_edges} edges...")
        logger.info(f"Using RMAT parameters: a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}, noise={noise}")

        # [NVIDIA CODE - Theta matrix construction]
        theta = np.array([[a, b], [c, d]])
        theta /= (a + b + c + d)  # Normalize

        # [NVIDIA REPOSITORY CODE] - Call with return_node_ids parameter
        # This is the proper way to track node coverage
        result = effective_nonsquare_rmat_exact(
            theta,
            num_edges,
            (num_nodes, num_nodes),
            noise_scaling=noise,
            batch_size=batch_size,
            dtype=np.int64,
            remove_selfloops=True,
            generate_back_edges=False,
            return_node_ids=1 if return_node_ids else 0,  # Track all nodes
        )

        if return_node_ids:
            edges_array, node_ids_array = result
            node_ids_set = set(node_ids_array.tolist())

            logger.info(f"Generated {len(edges_array)} edges")
            logger.info(
                f"Node coverage: {len(node_ids_set)}/{num_nodes} nodes used ({100 * len(node_ids_set) / num_nodes:.1f}%)")

            # [STANDALONE CODE - Convert to list of tuples for consistency]
            edges = [(int(src), int(dst)) for src, dst in edges_array]

            return edges, node_ids_set
        else:
            edges_array = result
            edges = [(int(src), int(dst)) for src, dst in edges_array]
            return edges


# ============================================================================
# [STANDALONE CODE] - Analysis and I/O utilities
# ============================================================================

def analyze_graph(edges: List[Tuple[int, int]], name: str = "Graph"):
    """
    [STANDALONE CODE]
    Analyze and print statistics about a graph
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"{name} Statistics")
    logger.info(f"{'=' * 60}")

    # Calculate degree statistics
    src_degrees = defaultdict(int)
    dst_degrees = defaultdict(int)
    all_nodes = set()

    for src, dst in edges:
        src_degrees[src] += 1
        dst_degrees[dst] += 1
        all_nodes.add(src)
        all_nodes.add(dst)

    out_degrees = list(src_degrees.values())
    in_degrees = list(dst_degrees.values())

    logger.info(f"Number of nodes: {len(all_nodes)}")
    logger.info(f"Number of edges: {len(edges)}")
    logger.info(f"Average out-degree: {np.mean(out_degrees):.2f}")
    logger.info(f"Average in-degree: {np.mean(in_degrees):.2f}")
    logger.info(f"Max out-degree: {max(out_degrees) if out_degrees else 0}")
    logger.info(f"Max in-degree: {max(in_degrees) if in_degrees else 0}")
    logger.info(f"{'=' * 60}\n")


def save_graph(edges: List[Tuple[int, int]], filename: str):
    """
    [STANDALONE CODE]
    Save graph edges to CSV file
    """
    df = pd.DataFrame(edges, columns=['source', 'target'])
    df.to_csv(filename, index=False)
    logger.info(f"Saved graph to {filename}")


def inject_missing_nodes(
        edges: List[Tuple[int, int]],
        node_ids_used: Set[int],
        total_nodes: int,
        strategy: str = "random_existing",
        seed: int = 42
) -> Tuple[List[Tuple[int, int]], Set[int]]:
    """
    [STANDALONE CODE - Post-processing for evaluation framework compatibility]

    Inject minimal edges for missing nodes to ensure complete node coverage.

    This is a post-processing step added for evaluation framework compatibility,
    not part of the core RMAT algorithm. It ensures the synthetic graph has
    exactly the same nodes as the original dataset for fair comparison.

    Args:
        edges: Original edge list from RMAT generation
        node_ids_used: Set of node IDs that appear in the generated edges
        total_nodes: Total number of nodes that should be in the final graph
        strategy: How to connect missing nodes
            - "random_existing": Connect each missing node to a random existing node
            - "random_pair": Add random edges between missing nodes and existing nodes
        seed: Random seed for reproducibility

    Returns:
        edges_with_injected: Edge list including injected edges
        complete_node_set: Set of all node IDs (should equal range(total_nodes))
    """
    np.random.seed(seed)

    missing_nodes = set(range(total_nodes)) - node_ids_used

    if not missing_nodes:
        logger.info("No missing nodes - complete coverage achieved!")
        return edges, node_ids_used

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Injecting Edges for Missing Nodes (Post-Processing)")
    logger.info(f"{'=' * 60}")
    logger.info(f"Missing nodes: {len(missing_nodes)}")
    logger.info(f"Strategy: {strategy}")

    edges_with_injected = edges.copy()
    existing_nodes = list(node_ids_used)

    injected_edges = []

    if strategy == "random_existing":
        # Connect each missing node to one random existing node
        # Add bidirectional edges to maintain undirected graph property
        for missing_node in sorted(missing_nodes):
            target_node = np.random.choice(existing_nodes)
            injected_edges.append((missing_node, target_node))
            injected_edges.append((target_node, missing_node))

    elif strategy == "random_pair":
        # Create random pairs between missing and existing nodes
        for missing_node in sorted(missing_nodes):
            target_node = np.random.choice(existing_nodes)
            injected_edges.append((missing_node, target_node))

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    edges_with_injected.extend(injected_edges)
    complete_node_set = node_ids_used | missing_nodes

    logger.info(f"Injected edges: {len(injected_edges)}")
    logger.info(f"Total edges (RMAT + injected): {len(edges_with_injected)}")
    logger.info(f"Node coverage after injection: {len(complete_node_set)}/{total_nodes}")
    logger.info(f"{'=' * 60}\n")

    return edges_with_injected, complete_node_set


def edges_to_adjacency_matrix(edges: List[Tuple[int, int]], num_nodes: int, is_directed: bool = True) -> np.ndarray:
    """
    Convert edge list to dense adjacency matrix

    Args:
        edges: List of (src, dst) tuples
        num_nodes: Size of the adjacency matrix (n x n)
        is_directed: Whether the graph is directed

    Returns:
        Dense adjacency matrix as numpy array
    """
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for src, dst in edges:
        adj_matrix[src, dst] = 1
        if not is_directed:
            adj_matrix[dst, src] = 1

    return adj_matrix


# ============================================================================
# [STANDALONE CODE] - Main execution pipeline
# ============================================================================

def main():
    """
    [STANDALONE CODE]
    Main execution pipeline
    """

    parser = argparse.ArgumentParser(
        description="RMAT Structure Generator for Cora Dataset with Poisoning Attack Support"
    )
    parser.add_argument(
        "--attack_method",
        type=str,
        choices=["CLGA", "random", "dice", "pgd", "minmax", "metattack", "nodeembeddingattack", "none"],
        default="none",
        help="Poisoning attack method (default: none for clean graph)"
    )
    parser.add_argument(
        "--attack_rate",
        type=float,
        choices=[0.01, 0.05, 0.10],
        default=0.10,
        help="Attack rate used in the graph poisoning attack (default: 0.10)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help="Dataset name (default: Cora)"
    )
    parser.add_argument(
        "--poisoned_dir",
        type=str,
        default="/home/luy25/robustsyntheticgraph/CLGA/poisoned_adj",
        help="Directory containing poisoned adjacency matrices (default: /home/luy25/robustsyntheticgraph/CLGA/poisoned_adj)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory for generated graphs (default: ./output)"
    )
    parser.add_argument(
        "--inject_missing_nodes",
        action="store_true",
        help="Inject edges for missing nodes to ensure complete node coverage (post-processing for evaluation framework compatibility)"
    )
    parser.add_argument(
        "--injection_strategy",
        type=str,
        choices=["random_existing", "random_pair"],
        default="random_existing",
        help="Strategy for injecting missing nodes (default: random_existing)"
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("RMAT Structure Generator for Cora Dataset - v3")
    logger.info("Core: NVIDIA's RMAT algorithm with return_node_ids")
    if args.inject_missing_nodes:
        logger.info(f"Post-processing: Node injection enabled ({args.injection_strategy})")
    else:
        logger.info("Post-processing: Node injection disabled")
    logger.info("=" * 60)

    # Step 1: Load original graph
    logger.info("\nStep 1: Loading Cora dataset...")

    if args.attack_method == "none":
        # Load clean graph
        logger.info("Loading clean Cora dataset...")
        loader = CoraDataLoader()
        original_edges, metadata = loader.load_graph()
        graph_type = "clean"
    else:
        # [CHANGE 4] Load poisoned graph
        logger.info(f"Loading poisoned graph ({args.attack_method}, rate={args.attack_rate})...")
        original_edges, metadata = load_poisoned_graph(
            attack_method=args.attack_method,
            attack_rate=args.attack_rate,
            dataset=args.dataset,
            poisoned_dir=args.poisoned_dir
        )
        graph_type = "poisoned"

    num_original_nodes = metadata['node_count']
    num_original_edges = metadata['edge_count']

    analyze_graph(original_edges, "Original Cora Graph")

    # Step 2: Fit RMAT generator
    logger.info("\nStep 2: Fitting RMAT generator...")
    generator = RMATGenerator(seed=42)
    generator.fit(original_edges, is_directed=True)

    # Step 3: Generate synthetic graph with node tracking
    logger.info("\nStep 3: Generating synthetic graph...")
    logger.info("Using return_node_ids=True for proper node coverage tracking")

    synthetic_edges, node_ids_used = generator.generate(
        num_nodes=num_original_nodes,
        num_edges=num_original_edges,
        is_directed=True,
        noise=0.5,
        return_node_ids=True,  # [KEY FEATURE] - Repository's node tracking
    )

    # Check node coverage
    missing_nodes = set(range(num_original_nodes)) - node_ids_used

    if missing_nodes:
        logger.warning(f"\n{'!' * 60}")
        logger.warning(f"Node Coverage Issue Detected:")
        logger.warning(f"  Total nodes expected: {num_original_nodes}")
        logger.warning(f"  Nodes in generated graph: {len(node_ids_used)}")
        logger.warning(f"  Missing nodes: {len(missing_nodes)}")
        logger.warning(f"{'!' * 60}")

        if args.inject_missing_nodes:
            logger.info("\n--inject_missing_nodes flag enabled: Adding edges for missing nodes...")
            synthetic_edges, node_ids_used = inject_missing_nodes(
                synthetic_edges,
                node_ids_used,
                num_original_nodes,
                strategy=args.injection_strategy,
                seed=42
            )
            missing_nodes = set(range(num_original_nodes)) - node_ids_used

            if not missing_nodes:
                logger.info(f"✓ Successfully achieved complete node coverage!")
            else:
                logger.error(f"✗ Node injection failed - still missing {len(missing_nodes)} nodes")
        else:
            logger.warning("\nTo achieve complete node coverage, run with --inject_missing_nodes")
            logger.warning("This is expected behavior with RMAT:")
            logger.warning("- RMAT's stochastic nature means some nodes may not be selected")
            logger.warning("- For feature alignment, you may need to:")
            logger.warning("  1. Add edges for missing nodes (--inject_missing_nodes), OR")
            logger.warning("  2. Filter features to only used nodes, OR")
            logger.warning("  3. Generate more edges to increase coverage")
            logger.warning(f"{'!' * 60}\n")
    else:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Perfect Node Coverage: All {num_original_nodes} nodes present!")
        logger.info(f"{'=' * 60}\n")

    analyze_graph(synthetic_edges, "Synthetic Cora Graph")

    # Step 4: Save results
    logger.info("\nStep 4: Saving results...")
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    # Convert to adjacency matrix
    logger.info("Converting to adjacency matrix...")
    adj_matrix = edges_to_adjacency_matrix(
        synthetic_edges,
        num_original_nodes,
        is_directed=True
    )
    logger.info(f"Adjacency matrix shape: {adj_matrix.shape}")

    if args.attack_method == "none":
        adj_filename = os.path.join(output_dir, "rmat_Cora.npy")
        edges_filename = os.path.join(output_dir, "rmat_Cora_edges.csv")
        metadata_filename = os.path.join(output_dir, "metadata.json")
    else:
        adj_filename = os.path.join(
            output_dir,
            f"rmat_Cora_{args.attack_method}_{args.attack_rate}.npy"
        )
        edges_filename = os.path.join(
            output_dir,
            f"rmat_Cora_{args.attack_method}_{args.attack_rate}_edges.csv"
        )
        metadata_filename = os.path.join(
            output_dir,
            f"metadata_{args.attack_method}_{args.attack_rate}.json"
        )
    # Save in format compatible with eval_graph.py
    # Following the pattern: {method}_{dataset}.npy
    np.save(adj_filename, adj_matrix)
    logger.info(f"Saved adjacency matrix to {adj_filename}")

    # Also save edge list for reference
    save_graph(synthetic_edges, edges_filename)

    # Save metadata including node coverage info
    metadata_output = {
        'original': {
            'nodes': num_original_nodes,
            'edges': num_original_edges
        },
        'synthetic': {
            'nodes': num_original_nodes,
            'edges': len(synthetic_edges),
            'nodes_used': len(node_ids_used),
            'node_coverage': len(node_ids_used) / num_original_nodes,
            'missing_nodes': len(missing_nodes),
        },
        'rmat_parameters': {
            'a': generator._fit_results[0],
            'b': generator._fit_results[1],
            'c': generator._fit_results[2],
            'd': generator._fit_results[3],
        },
        'generation_method': 'NVIDIA RMAT with node injection post-processing' if args.inject_missing_nodes else 'NVIDIA repository return_node_ids parameter',
        'node_injection': {
            'enabled': args.inject_missing_nodes,
            'strategy': args.injection_strategy if args.inject_missing_nodes else None,
            'complete_coverage': len(missing_nodes) == 0
        }
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata_output, f, indent=2)

    logger.info(f"Metadata saved to {output_dir}/metadata.json")

    logger.info("\n" + "=" * 60)
    logger.info("Generation Complete!")
    logger.info(f"Output files in: {output_dir}/")
    logger.info("=" * 60)

    return synthetic_edges, node_ids_used, missing_nodes


if __name__ == "__main__":
    main()