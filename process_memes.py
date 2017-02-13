import os
import random

data_dir = 'data/memes'
filepath = os.path.join(data_dir, 'cascades.txt')

random.seed(0)


def reformat():
    outpath = os.path.join(data_dir, 'cascades_processed.txt')
    with open(outpath, 'wb') as fo:
        with open(filepath, 'rb') as f:
            for line in f:
                line = line.replace(';', ' ').replace(',', ' ')
                chunks = line.split()
                chunks = [chunks[1]] + chunks[3:]
                if len(chunks) > 1:
                    fo.write(' '.join(chunks) + '\n')


def split_data():
    ratio = 0.75
    train_path = os.path.join(data_dir, 'train.txt')
    test_path = os.path.join(data_dir, 'test.txt')
    with open(train_path, 'wb') as f_train, open(test_path, 'wb') as f_test:
        with open(filepath, 'rb') as f:
            for line in f:
                if random.random() < ratio:
                    f_train.write(line)
                else:
                    f_test.write(line)


# reformat()
split_data()
