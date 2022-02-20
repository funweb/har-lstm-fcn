import os

import pandas as pd
from keras import backend as K
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, Flatten
from keras.layers import Input, Dense, LSTM, CuDNNLSTM, concatenate, Activation, GRU, SimpleRNN
from keras.models import Model

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model
from utils.layer_utils import AttentionLSTM


def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = LSTM(NUM_CELLS)(ip)
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

    # model.summary()

    # add load model code here to fine-tune

    return model


def generate_alstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

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

    # model.summary()

    # add load model code here to fine-tune

    return model


def train_val(epochs=2, batch_size=128):
    dataset_map = [('cairo_9999_0', 0),
                   ('cairo_9999_1', 1),
                   ('cairo_9999_2', 2),
                   ('cairo_999_0', 3),
                   ('cairo_999_1', 4),
                   ('cairo_999_2', 5),
                   ('cairo_1_0', 6),
                   ('cairo_1_1', 7),
                   ('cairo_1_2', 8),
                   ('cairo_2_0', 9),
                   ('cairo_2_1', 10),
                   ('cairo_2_2', 11),
                   ('cairo_3_0', 12),
                   ('cairo_3_1', 13),
                   ('cairo_3_2', 14),
                   ('cairo_4_0', 15),
                   ('cairo_4_1', 16),
                   ('cairo_4_2', 17),
                   ('cairo_5_0', 18),
                   ('cairo_5_1', 19),
                   ('cairo_5_2', 20),
                   ('milan_9999_0', 21),
                   ('milan_9999_1', 22),
                   ('milan_9999_2', 23),
                   ('milan_999_0', 24),
                   ('milan_999_1', 25),
                   ('milan_999_2', 26),
                   ('milan_1_0', 27),
                   ('milan_1_1', 28),
                   ('milan_1_2', 29),
                   ('milan_2_0', 30),
                   ('milan_2_1', 31),
                   ('milan_2_2', 32),
                   ('milan_3_0', 33),
                   ('milan_3_1', 34),
                   ('milan_3_2', 35),
                   ('milan_4_0', 36),
                   ('milan_4_1', 37),
                   ('milan_4_2', 38),
                   ('milan_5_0', 39),
                   ('milan_5_1', 40),
                   ('milan_5_2', 41),
                   ('kyoto7_9999_0', 42),
                   ('kyoto7_9999_1', 43),
                   ('kyoto7_9999_2', 44),
                   ('kyoto7_999_0', 45),
                   ('kyoto7_999_1', 46),
                   ('kyoto7_999_2', 47),
                   ('kyoto7_1_0', 48),
                   ('kyoto7_1_1', 49),
                   ('kyoto7_1_2', 50),
                   ('kyoto7_2_0', 51),
                   ('kyoto7_2_1', 52),
                   ('kyoto7_2_2', 53),
                   ('kyoto7_3_0', 54),
                   ('kyoto7_3_1', 55),
                   ('kyoto7_3_2', 56),
                   ('kyoto7_4_0', 57),
                   ('kyoto7_4_1', 58),
                   ('kyoto7_4_2', 59),
                   ('kyoto7_5_0', 60),
                   ('kyoto7_5_1', 61),
                   ('kyoto7_5_2', 62),
                   ('kyoto8_9999_0', 63),
                   ('kyoto8_9999_1', 64),
                   ('kyoto8_9999_2', 65),
                   ('kyoto8_999_0', 66),
                   ('kyoto8_999_1', 67),
                   ('kyoto8_999_2', 68),
                   ('kyoto8_1_0', 69),
                   ('kyoto8_1_1', 70),
                   ('kyoto8_1_2', 71),
                   ('kyoto8_2_0', 72),
                   ('kyoto8_2_1', 73),
                   ('kyoto8_2_2', 74),
                   ('kyoto8_3_0', 75),
                   ('kyoto8_3_1', 76),
                   ('kyoto8_3_2', 77),
                   ('kyoto8_4_0', 78),
                   ('kyoto8_4_1', 79),
                   ('kyoto8_4_2', 80),
                   ('kyoto8_5_0', 81),
                   ('kyoto8_5_1', 82),
                   ('kyoto8_5_2', 83),
                   ('kyoto11_9999_0', 85),
                   ('kyoto11_9999_1', 85),
                   ('kyoto11_9999_2', 86),
                   ('kyoto11_999_0', 87),
                   ('kyoto11_999_1', 88),
                   ('kyoto11_999_2', 89),
                   ('kyoto11_1_0', 90),
                   ('kyoto11_1_1', 91),
                   ('kyoto11_1_2', 92),
                   ('kyoto11_2_0', 93),
                   ('kyoto11_2_1', 94),
                   ('kyoto11_2_2', 95),
                   ('kyoto11_3_0', 96),
                   ('kyoto11_3_1', 97),
                   ('kyoto11_3_2', 98),
                   ('kyoto11_4_0', 99),
                   ('kyoto11_4_1', 100),
                   ('kyoto11_4_2', 101),
                   ('kyoto11_5_0', 102),
                   ('kyoto11_5_1', 103),
                   ('kyoto11_5_2', 104),
                   ]

    print("Num datasets : ", len(dataset_map))
    print()

    base_log_name = '%s_%d_cells_new_datasets.csv'
    base_weights_dir = '%s_%d_cells_weights/'

    MODELS = [
        ('lstmfcn', generate_lstmfcn),
        ('alstmfcn', generate_alstmfcn),
    ]

    # Number of cells
    CELLS = [8, 64, 128]

    # Normalization scheme
    # Normalize = False means no normalization will be done
    # Normalize = True / 1 means sample wise z-normalization
    # Normalize = 2 means dataset normalization.
    normalize_dataset = True

    for model_id, (MODEL_NAME, model_fn) in enumerate(MODELS):
        for cell in CELLS:
            successes = []
            failures = []

            if not os.path.exists(base_log_name % (MODEL_NAME, cell)):
                file = open(base_log_name % (MODEL_NAME, cell), 'w')
                file.write('%s,%s,%s,%s\n' % ('dataset_id', 'dataset_name', 'dataset_name_', 'test_accuracy'))
                file.close()

            for dname, did in dataset_map[0:2]:  # 约束运行数据集的个数

                MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[did]
                NB_CLASS = NB_CLASSES_LIST[did]

                # release GPU Memory
                K.clear_session()  # 释放内存

                file = open(base_log_name % (MODEL_NAME, cell), 'a+')

                weights_dir = base_weights_dir % (MODEL_NAME, cell)

                if not os.path.exists('weights/' + weights_dir):
                    os.makedirs('weights/' + weights_dir)

                dataset_name_ = weights_dir + dname

                # try:
                model = model_fn(MAX_SEQUENCE_LENGTH, NB_CLASS, cell)

                print('*' * 20, "Training model for dataset %s" % (dname), '*' * 20)

                # comment out the training code to only evaluate !
                train_model(model, did, dataset_name_, epochs=epochs, batch_size=batch_size, normalize_timeseries=normalize_dataset)

                acc = evaluate_model(model, did, dataset_name_, batch_size=batch_size, normalize_timeseries=normalize_dataset)

                s = "%d,%s,%s,%0.6f\n" % (did, dname, dataset_name_, acc)

                file.write(s)
                file.flush()

                successes.append(s)

                # except Exception as e:
                #     traceback.print_exc()
                #
                #     s = "%d,%s,%s,%s\n" % (did, dname, dataset_name_, 0.0)
                #     failures.append(s)
                #
                #     print()

                file.close()

            print('\n\n')
            print('*' * 20, "Successes", '*' * 20)
            print()

            result_csv_path = os.path.join("Results", "casas", dataset_name_.split("/")[0]+"_%s_%s.csv" % (epochs, batch_size))

            if os.path.exists(result_csv_path):
                df = pd.read_csv(result_csv_path, index_col=0)
            else:
                columns_list = ["data_name", "method", "acc"]
                df = pd.DataFrame(columns=columns_list, index=list(range(len(dataset_map))))

            for line in successes:
                l = line.split(",")
                df.loc[df.index == int(l[0]), "data_name"] = l[1]
                df.loc[df.index == int(l[0]), "method"] = l[2]
                df.loc[df.index == int(l[0]), "acc"] = l[3][:-1]

            print(df)
            df.to_csv(result_csv_path)

            print('\n\n')
            print('*' * 20, "Failures", '*' * 20)

            for line in failures:
                print(line)

    print("done...")


if __name__ == "__main__":
    train_val(epochs=2, batch_size=128)
