from typing import List, Dict, Union


def train_test_split(data, validate_size=0.3, random=False):
    """ Split dataset to training set & validation set

    Args:
        data: Inputs data of dataset
        validate_size: The ratio of validation set in dataset
        random: Randomly split dataset

    Returns:
        train: Inputs of training set
        validate: Inputs of validation set
    """

    train, validate = None, None

    if random:
        pass
        # TODO: Implement random split
    else:
        split = len(data) * (1 - validate_size)
        split = int(split)
        train = data[:split]
        validate = data[split:]

    return train, validate


def load_json(filename: str) -> Union[List, Dict]:
    import json

    with open(filename, 'r') as json_file:
        data = json.load(json_file)

    return data


def dump_json(filename: str, data: Union[List, Dict]) -> None:
    import json

    with open(filename, 'w') as json_file:
        json.dump(data, json_file)


def check_dir():
    import os
    import main
    work_dir = os.getcwd()
    file_dir = os.path.dirname(main.__file__)
    print('[  INFO  ] Current working dir: ', work_dir)
    if work_dir != file_dir:
        print('[  WARN  ] Not running in program dir, changing...')
        os.chdir(file_dir)
        print('[   OK   ] Now running in: ', os.getcwd())


def time_name():
    import datetime
    return datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
