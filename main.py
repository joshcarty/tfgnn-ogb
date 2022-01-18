import tensorflow as tf
import tensorflow_gnn as tfgnn

from tfgnn_ogb import data, model
from tfgnn_ogb.schema import TYPE_SPECS
from tfgnn_ogb.utils import ceildiv


def OgbnArxivModel(
    num_classes: int,
    convolution_size: int,
    residual_size: int,
    num_neighbourhoods: int,
    type_spec: tfgnn.GraphTensorSpec,
) -> tf.keras.Model:
    input = tf.keras.layers.Input(type_spec=type_spec)
    gnn = tfgnn.keras.ConvGNNBuilder(
        lambda edge: tfgnn.keras.layers.SimpleConvolution(
            tf.keras.layers.Dense(convolution_size)
        ),
        lambda node: tfgnn.keras.layers.ResidualNextState(
            tf.keras.layers.Dense(residual_size),
            activation=tf.keras.activations.relu,
        ),
    )
    hidden = gnn.Convolve()(input)
    for _ in range(num_neighbourhoods - 1):
        hidden = gnn.Convolve()(hidden)
    hidden = tfgnn.keras.layers.ReadoutFirstNode(node_set_name="paper")(hidden)
    output = tf.keras.layers.Dense(num_classes, activation="softmax")(hidden)
    return model.NodeClassificationModel(
        input, output, target_node="paper", label_name="label"
    )


def main() -> None:
    dataset = "ogbn-arxiv"
    neighbour_samples = (10, 2)
    batch_size = 64
    epochs = 50

    graph, splits, num_classes = data.ogb_as_networkx_graph(dataset)
    train_split, val_split = splits["train"], splits["valid"]
    type_spec = TYPE_SPECS[dataset]
    sampler = data.NodeSampler(graph, neighbour_samples)

    train_data = data.load_dataset_from_graph(
        graph,
        split=train_split,
        sampler=sampler,
        batch_size=batch_size,
        graph_type_spec=type_spec,
    )
    val_data = data.load_dataset_from_graph(
        graph,
        split=val_split,
        sampler=sampler,
        batch_size=batch_size,
        graph_type_spec=type_spec,
    )

    gnn = OgbnArxivModel(
        num_classes=num_classes,
        convolution_size=32,
        residual_size=128,
        num_neighbourhoods=sampler.num_neighbourhoods,
        type_spec=type_spec,
    )

    gnn.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(name="top_5", k=5),
        ],
    )

    gnn.fit(
        train_data,
        validation_data=val_data,
        steps_per_epoch=ceildiv(len(train_split), batch_size),
        validation_steps=ceildiv(len(val_split), batch_size),
        epochs=epochs,
    )


if __name__ == "__main__":
    main()
