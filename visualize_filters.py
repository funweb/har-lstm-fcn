from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, CuDNNLSTM, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST, TRAIN_FILES
from utils.generic_utils import load_dataset_at
from utils.keras_utils import visualize_filters
from utils.layer_utils import AttentionLSTM

import os
import traceback
import json
from keras import backend as K


def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = LSTM(NUM_CELLS)(ip)  # 一个输入
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)  # 另一个输入
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out, name="lstmfcn")

    model.summary()

    # add load model code here to fine-tune

    return model


def generate_attention_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(NUM_CELLS)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


if __name__ == '__main__':
    # COMMON PARAMETERS
    DATASET_ID = 0  # 更改
    num_cells = 8  # 更改
    model = generate_lstmfcn  # Select model to build  # 模型在这里设置

    # OLD 85 DATASET PARAMETERS
    # 自定义的 数据集 名字
    from all_datasets_trainingLY import dataset_map
    dataset_name = dataset_map[DATASET_ID][0]  # 'cbf'  # set to None to try to find out automatically for new datasets  设置为“无”以尝试自动查找新数据集

    # NEW 43 DATASET PARAMETERS
    model_name = 'lstmfcn'  # 更改名称

    # Visualizaion params
    CONV_ID = 0
    FILTER_ID = 1  # 不知道为啥 后面 assert 0 了, 只能大于0 < channel (C)

    """ <<<<< SCRIPT SETUP >>>>> """
    # Script setup
    sequence_length = MAX_SEQUENCE_LENGTH_LIST[DATASET_ID]
    nb_classes = NB_CLASSES_LIST[DATASET_ID]
    model = model(sequence_length, nb_classes, num_cells)

    if DATASET_ID >= 85:
        dataset_name = None

    if dataset_name is None:
        base_weights_dir = '%s_%d_cells_weights/'
        dataset_name = TRAIN_FILES[DATASET_ID][8:-6]
        weights_dir = base_weights_dir % (model_name, num_cells)

        dataset_name = weights_dir + dataset_name

    visualize_filters(model, DATASET_ID, dataset_name, conv_id=CONV_ID, filter_id=FILTER_ID, seed=0, normalize_timeseries=True)
