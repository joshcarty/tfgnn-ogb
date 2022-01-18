"""
Load Open Graph Benchmark data as GraphTensors and TensorFlow Datasets.

GraphTensors are created by sampling the neighbourhoods of each classification
node using NetworkX. Each node to be classified is stored as a separate
GraphTensor comprising of it and its n neighbourhoods. The number of neighbourhoods
to sample is a chosen parameter.
"""
import random
from dataclasses import dataclass
from typing import Any, Generator, Iterator, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from ogb.nodeproppred import NodePropPredDataset
from tqdm import tqdm


def ogb_as_networkx_graph(name: str) -> nx.Graph:
    """
    Load an Open Graph Benchmark dataset as a NetworkX graph.
    """
    dataset = NodePropPredDataset(name)
    splits = dataset.get_idx_split()
    ogb_graph, labels = dataset[0]

    num_nodes = ogb_graph["num_nodes"]
    ogb_features = ogb_graph["node_feat"]
    ogb_edgelist = ogb_graph["edge_index"]
    ogb_node_indices = np.arange(num_nodes)
    num_classes = labels.max() + 1

    graph = nx.from_edgelist(ogb_edgelist.T)
    data = zip(ogb_node_indices, ogb_features, labels)
    features = {
        node: {"features": features, "label": label}
        for node, features, label in data
    }
    nx.set_node_attributes(graph, values=features)
    return graph, splits, num_classes


@dataclass
class NodeSampler:
    graph: nx.Graph
    neighbour_samples: Tuple[int, ...] = (10, 2)

    @property
    def num_neighbourhoods(self) -> int:
        return len(self.neighbour_samples)

    def sample(self, seed_node: int) -> nx.Graph:
        sampled_nodes = {seed_node}
        to_sample = sampled_nodes.copy()
        for num_neighbours in self.neighbour_samples:

            for node in to_sample:
                neighbourhood_nodes = self.gather_neighbourhood(
                    node, exclude_nodes=sampled_nodes
                )
                if not neighbourhood_nodes:
                    continue
                sampled_neighbourhood_nodes = self.sample_neighbourhood(
                    neighbourhood_nodes, num_neighbours
                )
                sampled_nodes.update(sampled_neighbourhood_nodes)

            to_sample = sampled_neighbourhood_nodes.copy()

        return self.graph.subgraph(sampled_nodes)

    def gather_neighbourhood(self, node: int, exclude_nodes: set[int]) -> set[int]:
        neighbourhood_nodes = set(self.graph.neighbors(node))
        neighbourhood_nodes = self.exclude_already_sampled_nodes(
            neighbourhood_nodes, exclude_nodes
        )
        return neighbourhood_nodes

    @staticmethod
    def exclude_already_sampled_nodes(
        neighbourhood_nodes: Set[int], sampled_nodes: Set[int]
    ) -> set[int]:
        return neighbourhood_nodes - sampled_nodes

    @staticmethod
    def sample_neighbourhood(nodes: Set[int], num_neighbours: int) -> Set[int]:
        return set(random.choices(list(nodes), k=num_neighbours))


def _prepare_data_for_node_classification(
    graph: nx.Graph, seed_node: int
) -> List[Tuple[Any, Any]]:
    """
    Position seed node as the first node in the data.

    TensorFlow GNN has a convention whereby the node to be classified, the "seed node",
    is positioned first in the component. This is for use with layers such as
    `tfgnn.keras.layers.ReadoutFirstNode` which extracts the first node from a component.
    """
    seed_data = graph.nodes(data=True)[seed_node]
    data = [(seed_data["features"], seed_data["label"])]
    data += [
        (data["features"], data["label"])
        for node, data in graph.nodes(data=True)
        if node != seed_node
    ]
    return data


def generate_graph_samples(
    graph: nx.Graph, seed_nodes: Sequence[int], sampler: NodeSampler
) -> Iterator[tfgnn.GraphTensor]:
    """
    Lazily samples subgraphs from a NetworkX graph and converts them to
    GraphTensors.

    In practice, this would be a preprocessing step that builds the subgraphs
    using a Apache Beam, constructs the GraphTensors and serialises them as
    tf.Examples.
    """
    for seed_node in seed_nodes:
        subgraph = sampler.sample(seed_node)
        subgraph = nx.convert_node_labels_to_integers(
            subgraph, label_attribute="graph_index"
        )
        subgraph_seed_node = next(
            node
            for node, data in subgraph.nodes(data=True)
            if data["graph_index"] == seed_node
        )

        num_edges = subgraph.number_of_edges()
        edge_list = np.asarray(subgraph.edges)
        edges = tfgnn.EdgeSet.from_fields(
            sizes=[num_edges],
            adjacency=tfgnn.Adjacency.from_indices(
                source=("paper", edge_list[:, 0]),
                target=("paper", edge_list[:, 1]),
            ),
        )

        data = _prepare_data_for_node_classification(
            subgraph, subgraph_seed_node
        )
        features, labels = zip(*data)
        num_nodes = subgraph.number_of_nodes()
        nodes = tfgnn.NodeSet.from_fields(
            features={
                "hidden_state": np.asarray(features),
                "label": np.asarray(labels),
            },
            sizes=[num_nodes],
        )

        context = tfgnn.Context.from_fields(features=None)

        graph_tensor = tfgnn.GraphTensor.from_pieces(
            edge_sets={"cites": edges},
            node_sets={"paper": nodes},
            context=context,
        )

        yield graph_tensor


def merge_graph_batches(graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    """TensorFlow GNN expects batches of graphs to be a single component."""
    return graph.merge_batch_to_components()


def load_dataset_from_graph(
    graph: nx.Graph,
    split: Sequence[int],
    sampler: NodeSampler,
    batch_size: int,
    graph_type_spec: tfgnn.GraphTensorSpec,
) -> tf.data.Dataset:
    """
    Load a TensorFlow Dataset sampled as ego subgraphs from a NetworkX graph.
    
    Only suitable for small graphs or very low neighbourhood sizes. Since the
    Dataset is small, we can cache the data in memory. For larger neighbourhood
    sizes, it's preferable to write examples to disk using `main` and loading
    with `load_dataset_from_examples`.
    """

    def generator() -> Generator[tfgnn.GraphTensor, None, None]:
        """tf.data.Dataset expects a Callable that returns a Generator."""
        samples = generate_graph_samples(graph, split, sampler=sampler)
        yield from samples

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=graph_type_spec
    )
    return (
        dataset.cache()
        .repeat()
        .batch(batch_size, drop_remainder=True)
        .map(merge_graph_batches)
        .prefetch(tf.data.AUTOTUNE)
    )


def load_dataset_from_examples(
    path: str, batch_size: int, type_spec: tfgnn.GraphTensorSpec
) -> tf.data.TFRecordDataset:
    return (
        tf.data.TFRecordDataset(path)
        .batch(batch_size, drop_remainder=True)
        .map(lambda example: tfgnn.parse_example(type_spec, example))
        .repeat()
        .map(merge_graph_batches)
        .prefetch(tf.data.AUTOTUNE)
    )


def write_graph_tensors_to_examples(
    graph_generator: Iterator[tfgnn.GraphTensor], path: str
) -> None:
    # TODO: Shard over multiple files.
    with tf.io.TFRecordWriter(path) as writer:
        for graph in graph_generator:
            example = tfgnn.write_example(graph)
            writer.write(example.SerializeToString())


def main() -> None:
    dataset = "ogbn-arxiv"
    neighbour_samples = (10, 2)

    graph, splits, _ = ogb_as_networkx_graph(dataset)
    train_split, val_split = splits["train"], splits["valid"]
    sampler = NodeSampler(graph, neighbour_samples)

    for split_name, split in (("train", train_split), ("val", val_split)):
        generator = generate_graph_samples(graph, split, sampler)
        filename = f"{split_name}_hop_{'-'.join(map(str, neighbour_samples))}.tfrecords"
        write_graph_tensors_to_examples(
            graph_generator=tqdm(generator, total=len(split)),
            path=f"data/{dataset}/{split_name}/{filename}",
        )


if __name__ == "__main__":
    main()
