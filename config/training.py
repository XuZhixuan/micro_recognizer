config = {
    'define': './vgg8.json',
    'loss_function': {
        'name': 'MSELoss',
        'args': {}
    },
    'optimizer': {
        'name': 'SGD',
        'args': {
            'lr': 0.001
        }
    },
}
