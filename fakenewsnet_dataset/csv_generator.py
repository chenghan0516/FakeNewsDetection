import os
import json
import pandas as pd
import numpy as np


def get_all_news(target_dir, lable):
    output = pd.DataFrame({'title': [], 'text': [], 'lable': []})
    for (root, dirs, files) in target_dir:
        # print(root)
        # print(dirs)
        # print(files)
        if files == []:
            print(root)
        for file_name in files:
            if file_name == "news content.json":
                with open(os.path.join(root, file_name), "r") as f:
                    data = json.load(f)
                    new = pd.DataFrame({'title': [data['title']], 'text': [
                                       data['text']], 'lable': [lable]})
                    output = output.append(new, ignore_index=True)
        # print('--------------------------------')
    return output


def get_subject_news(subject):
    Real_files = os.walk(
        "C:\\Users\\Wu\\Desktop\\Casual Python\\fakenewsnet_dataset\\{}\\real".format(subject))
    Real_data = get_all_news(Real_files, 1)

    Fake_files = os.walk(
        "C:\\Users\\Wu\\Desktop\\Casual Python\\fakenewsnet_dataset\\{}\\fake".format(subject))
    Fake_data = get_all_news(Fake_files, 0)

    all_data = pd.concat(
        [Real_data, Fake_data], axis=0)
    all_data.to_csv("fakenewsnet_dataset\\{}.csv".format(subject), index=False)


if __name__ == '__main__':
    get_subject_news("politifact")
    get_subject_news("gossipcop")
