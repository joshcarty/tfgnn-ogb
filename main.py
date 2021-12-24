import tensorflow as tf

from tfgnn_ogb import model, data
from tfgnn_ogb.schema import TYPE_SPECS
from tfgnn_ogb.utils import ceildiv


def main() -> None:
    dataset = "ogbn-arxiv"
    neighbours = 2
    batch_size = 64
    epochs = 100

    _, splits, num_classes = data.ogb_as_networkx_graph(dataset)
    train_split, val_split = splits["train"], splits["valid"]
    type_spec = TYPE_SPECS[dataset]

    train_data = data.load_dataset_from_examples(
        path=f"data/{dataset}/train/train_hop_{neighbours}.tfrecords",
        batch_size=batch_size,
        type_spec=type_spec,
    )
    val_data = data.load_dataset_from_examples(
        path=f"data/{dataset}/val/val_hop_{neighbours}.tfrecords",
        batch_size=batch_size,
        type_spec=type_spec,
    )

    gnn = model.build_model(num_classes, type_spec)

    gnn.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(name="top_5", k=5),
        ],
        run_eagerly=False,
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
