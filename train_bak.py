import os

import numpy as np
import tensorflow as tf

import env
import model

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main():
    train_writer = tf.summary.create_file_writer("logs/train")
    test_writer = tf.summary.create_file_writer("logs/test")
    m = model.get_model()
    step = tf.Variable(0, trainable=False)

    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
        [2000, 5000, 10000], [1e-3, 2e-4, 1e-4, 0])
    lr = 1. * schedule(step)

    optimizer = tf.keras.optimizers.Adam(lr)
    ckpt = tf.train.Checkpoint(step=step, optimizer=optimizer, net=m)
    manager = tf.train.CheckpointManager(ckpt, './checkpoints', max_to_keep=30)

    @tf.function
    def train_step(data, label):
        with tf.GradientTape() as tape:
            logit = m((data, .5))
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit, 1), tf.argmax(label, 1)), tf.float32))
        gradients = tape.gradient(loss, m.trainable_variables)
        optimizer.apply_gradients(zip(gradients, m.trainable_variables))
        return loss, accuracy

    @tf.function
    def test_evaluate(data, label):
        logit = m((data, 0.))
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit)
        accuracies = tf.cast(tf.equal(tf.argmax(logit, 1), tf.argmax(label, 1)), tf.float32)

        return losses, accuracies

    @tf.function
    def test(data, label):
        logit = m((data, 0.))
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logit))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logit, 1), tf.argmax(label, 1)), tf.float32))

        return loss, accuracy

    def evaluate(eva_dataset: Dataset):
        losses = []
        accuracies = []
        batches = eva_dataset.get_all_batches()
        for batch in batches:
            losses_batch, accuracies_batch = test_evaluate(batch['data'], batch['label'])
            for loss in losses_batch.numpy():
                losses.append(loss)
            for acc in accuracies_batch.numpy():
                accuracies.append(acc)

        return float(np.mean(accuracies))

    print("\n======== classes ========")
    for i, j in enumerate(env.classes):
        print("  %d:" % i, j)
    print("=========================\n")

    print(m.summary())
    dataset = Dataset().load("./dataset/spectrogram/train").shuffle()
    test_dataset = Dataset().load("./dataset/spectrogram/test").shuffle()
    print("training dataset: %d items." % len(dataset.queue))
    print("testing  dataset: %d items." % len(test_dataset.queue))
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    print("start training.. \n\n\n")
    i = 0
    acc_max = 0.
    while i < 10005:
        i = ckpt.step.numpy()
        ckpt.step.assign_add(1)
        data, label = dataset.next()

        loss, acc = train_step(data, label)

        if i % 100 == 0:
            test_data, test_label = test_dataset.next()
            test_loss, test_acc = test(test_data, test_label)

            with train_writer.as_default():
                tf.summary.scalar("fig/loss", loss.numpy(), step=i)
                tf.summary.scalar("fig/acc", acc.numpy(), step=i)
            with test_writer.as_default():
                tf.summary.scalar("fig/loss", test_loss.numpy(), step=i)
                tf.summary.scalar("fig/acc", test_acc.numpy(), step=i)

        if i % 200 == 0:
            acc = evaluate(test_dataset)
            acc_max = max(acc_max, acc)
            print("[step: %d]" % i, "Accuracy: %.5f" % (acc * 100) + "%,", ("Best: %.5f" % (acc_max * 100)) + "%")

            save_path = manager.save()
            # print("Saved checkpoint for step {}: {}".format(i, save_path))


if __name__ == '__main__':
    main()
