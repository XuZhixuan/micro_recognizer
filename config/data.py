from Source import *

config = {
    'bound': {
        'low': 16.0,
        'inter': 12.0
    },
    'source': SavedSource,
    'kwargs': {
        'dir_name': './storage/saves/20210509T154546/'
    },
    'cache': './storage/cache/'
    # 'source': DBSource,
    # 'kwargs': {
    #     'db_hostname': '49.208.46.17',
    #     'db_password': '123456',
    #     'db_username': 'root',
    #     'db_name': 'mtrl_info_fg',
    #     'base_url': 'http://49.208.46.17:3000'
    # }
}
