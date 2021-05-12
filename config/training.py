config = {
    'define': [
        {
            "class": "BatchNorm2d",
            "args": {
                "num_features": 1
            }
        },
        {
            "class": "Conv2d",
            "args": {
                "in_channels": 1,
                "out_channels": 64,
                "kernel_size": 5
            }
        },
        {
            "class": "ReLU",
            "args": {
                "inplace": True
            }
        },
        {
            "class": "BatchNorm2d",
            "args": {
                "num_features": 64
            }
        },
        {
            "class": "Conv2d",
            "args": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3
            }
        },
        {
            "class": "ReLU",
            "args": {
                "inplace": True
            }
        },
        {
            "class": "MaxPool2d",
            "args": {
                "kernel_size": 2
            }
        },
        {
            "class": "BatchNorm2d",
            "args": {
                "num_features": 128
            }
        },
        {
            "class": "Conv2d",
            "args": {
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": 3
            }
        },
        {
            "class": "ReLU",
            "args": {
                "inplace": True
            }
        },
        {
            "class": "BatchNorm2d",
            "args": {
                "num_features": 256
            }
        },
        {
            "class": "Conv2d",
            "args": {
                "in_channels": 256,
                "out_channels": 512,
                "kernel_size": 3
            }
        },
        {
            "class": "ReLU",
            "args": {
                "inplace": True
            }
        },
        {
            "class": "MaxPool2d",
            "args": {
                "kernel_size": 2
            }
        },
        {
            "class": "AdaptiveAvgPool2d",
            "args": {
                "output_size": 20
            }
        },
        {
            "class": "Flatten",
            "args": {}
        },
        {
            "class": "Linear",
            "args": {
                "in_features": 204800,
                "out_features": 1024
            }
        },
        {
            "class": "ReLU",
            "args": {
                "inplace": True
            }
        },
        {
            "class": "Linear",
            "args": {
                "in_features": 1024,
                "out_features": 1024
            }
        },
        {
            "class": "ReLU",
            "args": {
                "inplace": True
            }
        },
        {
            "class": "Linear",
            "args": {
                "in_features": 1024,
                "out_features": 502
            }
        }
    ],
    'loss_function': {
        'name': 'CrossEntropyLoss',
        'args': {}
    },
    'optimizer': {
        'name': 'SGD',
        'args': {
            'lr': 0.001
        }
    },
    'epochs': 15
}
