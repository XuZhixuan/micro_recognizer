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
