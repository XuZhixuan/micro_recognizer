from torchsummary import summary
from Modules import *


from Source import *


def debug():
    # source = DBSource(
    #     db_hostname='49.208.46.17',
    #     db_password='123456',
    #     db_username='root',
    #     db_name='mtrl_info_fg',
    #     base_url='http://49.208.46.17:3000'
    # )

    source = SavedSource('data.pkl')
    for datum in source:
        print('%s found' % datum.path)
    pass


if __name__ == '__main__':
    debug()
