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
