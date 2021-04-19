def train_test_split(x_data, y_data, validate_size=0.3, random=False):
    """ Split dataset to training set & validation set

    Args:
        x_data: Inputs data of dataset
        y_data: Tags of dataset
        validate_size: The ratio of validation set in dataset
        random: Randomly split dataset

    Returns:
        x_train: Inputs of training set
        x_validate: Inputs of validation set
        y_train: Tags of training set
        y_validate: Tags of validation set
    """
    if len(x_data) != len(y_data):
        print("[ Error ] the size of data & tags don't match")
        exit(1)

    x_train, y_train, x_validate, y_validate = None, None, None, None

    if random:
        pass
        # TODO: Implement random split
    else:
        split = len(x_data) * (1 - validate_size)
        split = int(split)
        x_train = x_data[:split]
        x_validate = x_data[split:]

        y_train = y_data[:split]
        y_validate = y_data[split:]

    return x_train, x_validate, y_train, y_validate


def transform_manifest(results_file, manifest_file, num=1):
    """
    加载清单转换文件，将结果文件转换为加载清单
    Args:
        results_file: 结果文件路径
        manifest_file: 加载列表文件路径，新内容将追加在文件末尾
        num: 起始文件编号，默认为1

    Returns:

    """
    with open(results_file, 'r') as results:
        with open(manifest_file, 'a') as manifest:
            while True:
                line = results.readline()
                if line:
                    new_line = str(num) + ' ' + line
                    manifest.write(new_line)
                    num += 1
                else:
                    break


def unwrap_dataset(filename: str, dir_name: str = ''):
    import zipfile 

    file = zipfile.ZipFile(filename)


def load_raw_dataset(dir_name: str):
    from Tools import ImageLoader


def load_dataset(filename: str = './dataset.pkl'):
    import pickle
    with open(filename, 'rb') as bin_file:
        dataset = pickle.load(bin_file)
    return dataset


def save_dataset(dataset: list, filename: str = './dataset.pkl'):
    import pickle
    with open(filename, 'wb') as bin_file:
        pickle.dump(dataset, bin_file)
    pass
