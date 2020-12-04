import os
import random


def split(train_num):
    if os.path.exists('train.txt'):
        os.remove('train.txt')
    if os.path.exists('val.txt'):
        os.remove('val.txt')

    with open('train.txt', 'a') as t:
        with open('val.txt', 'a') as v:
            with open('trainval.txt', 'r') as f:
                txt = f.readlines()
                image_inds = [line.strip() for line in txt]
                train = random.sample(image_inds, train_num)
                train.sort()
                for d in train:
                    t.write(str(d + "\n"))
                for d in image_inds:
                    if d not in train:
                        v.write(str(d + "\n"))
