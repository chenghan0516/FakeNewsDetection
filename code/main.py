import globals
from util import preprocess
import train

do_preprocess = 0
do_train = 1
do_evaluate = 0

if __name__ == "__main__":

    globals.init()

    if do_preprocess:
        print("Start preprocess")
        preprocess()
    if do_train:
        print("Start training")
        train.train()
    if do_evaluate:
        print("Start evaluating")
