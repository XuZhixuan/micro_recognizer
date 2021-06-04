from Source import *

config = {
    'bound': {
        'low': 15.0,
        'inter': 25.0
    },
    'source': FileSource,
    'kwargs': {
        'dir_name': './storage/zips/'
    },
    'num_class': 20,
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
