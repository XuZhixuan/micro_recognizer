from torchsummary import summary
from tensorboardX import SummaryWriter

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

    source = FileSource('zips')

    # source = SavedSource('data.pkl')
    source.dump('data0.pkl')


if __name__ == '__main__':
    debug()
