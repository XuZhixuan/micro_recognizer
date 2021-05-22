config = {
    'define': [
        {
            "class": "Conv2d",
            "args": {
                "in_channels": 1,
                "out_channels": 64,
                "kernel_size": 5,
                "stride": 2
            }
        },
        # {
        #     "class": "BatchNorm2d",
        #     "args": {
        #         "num_features": 64
        #     }
        # },
        {
            "class": "LeakyReLU",
            "args": {
                "inplace": True
            }
        },

        # {
        #     "class": "Conv2d",
        #     "args": {
        #         "in_channels": 64,
        #         "out_channels": 64,
        #         "kernel_size": 3
        #     }
        # },
        # {
        #     "class": "BatchNorm2d",
        #     "args": {
        #         "num_features": 64
        #     }
        # },
        # {
        #     "class": "LeakyReLU",
        #     "args": {
        #         "inplace": True
        #     }
        # },
        {
            "class": "MaxPool2d",
            "args": {
                "kernel_size": 2
            }
        },
        {
            "class": "Conv2d",
            "args": {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "stride": 2
            }
        },
        # {
        #     "class": "BatchNorm2d",
        #     "args": {
        #         "num_features": 128
        #     }
        # },
        {
            "class": "LeakyReLU",
            "args": {
                "inplace": True
            }
        },
        # {
        #     "class": "Conv2d",
        #     "args": {
        #         "in_channels": 128,
        #         "out_channels": 128,
        #         "kernel_size": 3
        #     }
        # },
        # {
        #     "class": "BatchNorm2d",
        #     "args": {
        #         "num_features": 128
        #     }
        # },
        # {
        #     "class": "LeakyReLU",
        #     "args": {
        #         "inplace": True
        #     }
        # },
        {
            "class": "MaxPool2d",
            "args": {
                "kernel_size": 2
            }
        },
        {
            "class": "Conv2d",
            "args": {
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": 3,
                "stride": 2
            }
        },
        # {
        #     "class": "BatchNorm2d",
        #     "args": {
        #         "num_features": 256
        #     }
        # },
        {
            "class": "LeakyReLU",
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
            "class": "Flatten",
            "args": {}
        },
        {
            "class": "Linear",
            "args": {
                "in_features": 2304,
                "out_features": 512
            }
        },
        {
            "class": "LeakyReLU",
            "args": {
                "inplace": True
            }
        },
        {
            "class": "Linear",
            "args": {
                "in_features": 512,
                "out_features": 1
            }
        }
    ],
    'loss_function': {
        'name': 'MSELoss',
        'args': {}
    },
    'optimizer': {
        'name': 'Adam',
        'args': {
            'lr': 0.004
        }
    },
    'epochs': 500
}
