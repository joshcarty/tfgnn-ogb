"""
Load Open Graph Benchmark data as GraphTensors and TensorFlow Datasets.

GraphTensors are created by sampling the neighbours of each classification
node using NetworkX. Each node to be classified is stored as a separate
GraphTensor comprising of it and its n neighbours. The number of neighbours
to sample is a chosen parameter.

Sampling a node's immediate neighbourhood and is relatively fast. You can
lazily load them to a tf.data.Dataset at training time with the
`load_dataset_from_graph` function. The GraphTensors are then cached in
memory making subsequent epochs very fast.

Sampling more than an node's immediate neighbours is slow and consumes a lot
pof memory. It is best treated as a preprocessing step. You can use
`generate_graph_samples` with the `write_graph_tensors_to_examples` function
to write the TFRecords to disk and the `load_dataset_from_examples` function
to load them as a tf.data.Dataset.

For reference, sampling 2 hop neighbourhoods for ogbn-arxiv took 6h24 for the
training set and 6h10 for the validation set on a 2018 MacBook Pro. The resulting
TFRecords were 83GB and 85GB respectively.
"""

from typing import Any, Generator, Iterator, List, Sequence, Tuple

import networkx as nx
import numpy as np
import tensorflow_gnn as tfgnn
import tensorflow as tf
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
    graph: nx.Graph, seed_nodes: Sequence[int], neighbours: int = 2
) -> Iterator[tfgnn.GraphTensor]:
    """
    Lazily samples subgraphs from a NetworkX graph and converts them to
    GraphTensors.

    In practice, this would be a preprocessing step that builds the subgraphs
    using a Apache Beam, constructs the GraphTensors and serialises them as
    tf.Examples.
    """
    for seed_node in seed_nodes:
        subgraph = nx.ego_graph(graph, seed_node, radius=neighbours)
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

        graph_tensor = tfgnn.GraphTensor.from_pieces(
            edge_sets={"cites": edges}, node_sets={"paper": nodes}
        )

        yield graph_tensor


def merge_graph_batches(graph: tfgnn.GraphTensor) -> tfgnn.GraphTensor:
    """TensorFlow GNN expects batches of graphs to be a single component."""
    return graph.merge_batch_to_components()


def load_dataset_from_graph(
    graph: nx.Graph,
    split: Sequence[int],
    neighbours: int,
    batch_size: int,
    graph_type_spec: tfgnn.GraphTensorSpec,
) -> tf.data.Dataset:
    """
    Load a TensorFlow Dataset sampled as ego subgraphs from a NetworkX graph.
    
    Only suitable for small graphs or very low neighbourhood sizes e.g. neighbours=1.
    Since the Dataset is small and sampling is slow, we cache the data in memory.
    """

    def generator() -> Generator[tfgnn.GraphTensor, None, None]:
        """tf.data.Dataset expects a Callable that returns a Generator."""
        samples = generate_graph_samples(graph, split, neighbours=neighbours)
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
    neighbours = 1

    graph, splits, _ = ogb_as_networkx_graph(dataset)
    train_split, val_split = splits["train"], splits["valid"]

    for split_name, split in (('train', train_split), ('val', val_split)):
        generator = generate_graph_samples(graph, split, neighbours)
        write_graph_tensors_to_examples(
            graph_generator=tqdm(generator,total=len(split)),
            path=f"data/{dataset}/{split_name}/{split_name}_hop_{neighbours}.tfrecords"
        )


if __name__ == '__main__':
    main()
