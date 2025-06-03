import os
import json
import numpy as np
import tensorflow as tf
policy = tf.keras.mixed_precision.Policy('float32')
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import layers

from ..model.losses import MyWeightedBinaryCrossEntropy
from ..model.metrics import MyMeanDSC, MyMeanIoU, MyJaccardIndex, MyF1Score, MyF2Score
from ..model.utils import crop_labels_to_shape
from ..model.utils import read_tfrecord, set_tile_shape, filter_empty_labels, to_numpy_or_python_type, random_flip_and_rotation, random_image_adjust, keep_fraction_of_empty_labels
from ..model.custom_objects import custom_objects

from ..model.config import config

class PersistentBestModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, **kwargs):

        if 'save_best_only' in kwargs.keys() and not kwargs.get('save_best_only'):
            print('Warning: The PersistentBestModelCheckpoint is specifically for saving best models accross runs.')
            print('         Will ignore the provided `save_best_only=False` option.')
        kwargs.update(dict(save_best_only=True))

        super().__init__(filepath, **kwargs)

        assert self.save_freq == 'epoch'

        if os.path.exists(filepath):
            assert os.path.isdir(filepath)
        else:
            os.makedirs(filepath)

        self.json_path = os.path.join(filepath, 'best_logs.json')
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as file:
                best_logs = json.load(file)

            self.best = best_logs.get(self.monitor)

    
    def on_epoch_end(self, epoch, logs=None):
        try:
            logs = logs or {}
            logs = to_numpy_or_python_type(logs)
            current = logs.get(self.monitor)
            if self.monitor_op(current, self.best):
                with open(self.json_path, 'w') as file:
                    json.dump(logs, file)

        finally:
            super().on_epoch_end(epoch, logs)


def get_conv_block(x, filters, kernel_size, dropout_rate, activation, stage):
    x = layers.Conv2D( filters, kernel_size,  name=f'{stage}_conv1', activation=activation)(x)
    if dropout_rate > 0.:
        x = layers.SpatialDropout2D(dropout_rate, name=f'{stage}_drop1')(x)
    x = layers.BatchNormalization(            name=f'{stage}_bnor1')(x)
    x = layers.Conv2D( filters, kernel_size,  name=f'{stage}_conv2', activation=activation)(x)
    if dropout_rate > 0.:
        x = layers.SpatialDropout2D(dropout_rate, name=f'{stage}_drop2')(x)
    x = layers.BatchNormalization(            name=f'{stage}_bnor2')(x)
    return x


def get_upconv(x, copy, filters, activation, stage):
    x = layers.Conv2DTranspose(filters, (2, 2), 2, name=f'{stage}_upsmp', activation=activation)(x)
    ver_pad = (copy.shape[1] - x.shape[1]) // 2
    hor_pad = (copy.shape[2] - x.shape[2]) // 2
    paste = layers.Cropping2D( (ver_pad, hor_pad), name=f'{stage}_paste')(copy)
    x = layers.Concatenate(                        name=f'{stage}_input')([paste, x])
    return x


def get_model(tile_shape, channels, layer_depth=5, filters_root=16, activation = 'relu'):

    kernel_size = 3

    filters_list = [filters_root * 2 ** i for i in range(layer_depth)]
    level_list = list(range(1, len(filters_list) + 1))
    copy_list = []

    for l, filters in zip(level_list, filters_list):
        if l == 1:
            x = input = tf.keras.Input(tile_shape + (channels,), name='contract1_input', dtype='float32')
            x = layers.GaussianNoise(0.01)(x)
        else:
            x = layers.MaxPooling2D(     (2, 2), name=f'contract{l}_input', strides=2)(x)

        x = get_conv_block(x, filters, kernel_size, dropout_rate=.20, activation=activation, stage=f'contract{l}')
        copy_list.append(x)

    for l, filters, copy in zip(level_list[::-1], filters_list[::-1], copy_list[::-1]):
        if l == max(level_list):
            continue

        x = get_upconv(x, copy, filters, activation, stage=f'expand{l}')

        x = get_conv_block(x, filters, kernel_size, dropout_rate=.10, activation=activation, stage=f'expand{l}')

    x = layers.Conv2D(  1, 1, name='final_conv')(x)
    output = layers.Activation("sigmoid", dtype='float32', name='output')(x)

    # Define the model
    model = tf.keras.Model(input, output)

    return model


def get_dataset(split, fraction_of_empty_to_keep):

    files = tf.data.Dataset.list_files(f'./{split}/*.tfrecord')
    return (
        files
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        .map(set_tile_shape(tile_shape), num_parallel_calls=tf.data.AUTOTUNE)
        # .filter(keep_fraction_of_empty_labels(fraction_of_empty_to_keep))
        .cache()
        # .shuffle(buffer_size=32, reshuffle_each_iteration=True)
        .map(random_flip_and_rotation, num_parallel_calls=tf.data.AUTOTUNE)
        .map(random_image_adjust, num_parallel_calls=tf.data.AUTOTUNE)
        .map(crop_labels_to_shape(window_shape), num_parallel_calls=tf.data.AUTOTUNE))


def get_test_dataset():

    files = tf.data.Dataset.list_files(f'../tst/*.tfrecord')
    return (
        files
        .interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
        .map(read_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        .map(set_tile_shape(tile_shape), num_parallel_calls=tf.data.AUTOTUNE)
        .map(crop_labels_to_shape(window_shape), num_parallel_calls=tf.data.AUTOTUNE))


if __name__  == '__main__':


    activation = tf.nn.leaky_relu
    loss_pos_weight = 2
    n_epochs_per_run = 1200

    basedir, kf_id = os.path.split(os.getcwd().replace('..', ''))
    organelle = os.path.basename(basedir)

    if not kf_id.startswith('kf'):
        print('Error: this script is meant to be run inside a k-fold directory. Exiting...')
        exit()

    if organelle not in config.keys():
        print('Error: Could not find the organelle name from the current working directory. Exiting...')
        exit()

    params = config[organelle]
    tile_shape = params['tile_shape']
    layer_depth = params['layer_depth']
    window_shape = params['window_shape']
    batch_size = params['batch_size']
    filters_root = params['filters_root']
    fraction_of_empty_to_keep = params['fraction_of_empty_to_keep']

    train_dataset = get_dataset('tra', fraction_of_empty_to_keep=fraction_of_empty_to_keep)
    validation_dataset = get_dataset('val', fraction_of_empty_to_keep=fraction_of_empty_to_keep)

    if os.path.exists('./ckpt/last/saved_model.pb'):
        with open('./logs/metrics.tsv') as file:
            initial_epoch = sum(1 for line in file) - 1
        model = tf.keras.models.load_model('ckpt/last', custom_objects=custom_objects)
    else:
        initial_epoch = 0
        #building the model
        model = get_model(tile_shape, channels=1, layer_depth=layer_depth, filters_root=filters_root, activation=activation)

        loss = MyWeightedBinaryCrossEntropy(pos_weight=loss_pos_weight)
        mean_iou_metric = MyMeanIoU(name='mean_iou', threshold=0.5)
        mean_dsc_metric = MyMeanDSC(name='dice_coefficient', threshold=0.5)
        jc_index_metric = MyJaccardIndex(name='jaccard_index', threshold=0.5)
        f1_score_metric = MyF1Score(name='f1_score', threshold=0.5)
        f2_score_metric = MyF2Score(name='f2_score', threshold=0.5)

        model.compile(
            loss=loss,
            optimizer='adam',
            metrics=[
                mean_iou_metric,
                mean_dsc_metric,
                'Recall',
                jc_index_metric,
                f1_score_metric,
                f2_score_metric,
            ],
            jit_compile=False,
        )

    total_epochs = n_epochs_per_run * (initial_epoch // n_epochs_per_run + 1)

    history = model.fit(
        train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
        validation_data=validation_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
        initial_epoch=initial_epoch,
        epochs=total_epochs,
        callbacks=[
            tf.keras.callbacks.CSVLogger('./logs/metrics.tsv', separator='\t', append=True),
            tf.keras.callbacks.TensorBoard('./logs', profile_batch=0),
            tf.keras.callbacks.ModelCheckpoint('./ckpt/last'),
            PersistentBestModelCheckpoint('./ckpt/best_loss', save_best_only=True, monitor='val_loss', verbose=1),
            PersistentBestModelCheckpoint('./ckpt/best_dice', save_best_only=True, monitor='val_dice_coefficient', mode='max', verbose=1),
            PersistentBestModelCheckpoint('./ckpt/best_f2', save_best_only=True, monitor='val_f2_score', mode='max', verbose=1),
        ],
        verbose=2,
        workers=12,
    )
    final_epoch = initial_epoch + len(history.history['loss'])
    model.save(f'./ckpt/intermediate/{final_epoch:05d}')

    test_dataset = get_test_dataset()
    thresholds = np.arange(0, 1, 0.005).tolist()
    metrics = [
        tf.keras.metrics.TruePositives(thresholds=thresholds),
        tf.keras.metrics.FalsePositives(thresholds=thresholds),
        tf.keras.metrics.TrueNegatives(thresholds=thresholds),
        tf.keras.metrics.FalseNegatives(thresholds=thresholds),
    ]
    model.compile(metrics=metrics)
    result = model.evaluate(test_dataset.batch(8), verbose=2)[1:]
    result =  [r.astype(int).tolist() for r in result]

    json_path = os.path.join("evaluation", f"{final_epoch:05d}_evaluation.json")
    with open(json_path, "w") as file:
        json.dump(result, file)
