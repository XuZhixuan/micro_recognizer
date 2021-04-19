import pickle
import zipfile
import re
import os

from abc import abstractmethod

from Tools import ImageLoader


class Source:
    def __init__(self):
        self.data = []
        self.length = 0

    @abstractmethod
    def fetch(self):
        raise NotImplementedError

    def __iter__(self):
        self.curr = 0
        return self

    def __getitem__(self, item):
        return self.data[item]

    def __next__(self):
        if self.curr >= self.length:
            raise StopIteration
        else:
            index = self.curr
            self.curr += 1
            return self.data[index]

    def __add__(self, other):
        self.data += other
        return self

    def dump(self, path: str):
        with open(path, 'wb') as bin_file:
            pickle.dump(self.data, bin_file)


class FileSource(Source):
    def __init__(self, dir_name: str):
        super().__init__()
        self.dir_name = dir_name
        self.fetch()

    def fetch(self):
        # Detect the zip files presented in given dir
        zip_files = []
        for file in os.listdir(self.dir_name):
            if re.match(r'(.*)\.zip', file):
                zip_files.append(
                    zipfile.ZipFile(self.dir_name + '/' + file)
                )

        data = []
        # Load data from each zip file
        for zip_file in zip_files:
            data += self.load(zip_file)

        self.data = data

    @staticmethod
    def load(file: zipfile.ZipFile) -> list:
        nums = []
        # Determine the first file in zip file
        for name in file.namelist():
            # Extract every file from zip file
            file.extract(name, path='images')
            num = re.search(r'GS(.*)\.png', name)
            if num:
                nums.append(
                    int(num.group(1))
                )
        begin = min(nums)
        # Determine the name of extract dir
        dir_name = file.namelist()[0].split('/')[0]

        # Transform the result file into a load list
        with open('images/' + dir_name + '/result.txt', 'r') as results_file:
            results = results_file.readlines()

        manifest = []
        for result in results:
            result = result.split()
            manifest.append((
                file.filename + ':GS' + str(begin) + '.png',
                'images/' + dir_name + '/GS' + str(begin) + '.png',
                'images/' + dir_name + '/RGB' + str(begin) + '.png',
                float(result[0]),
                float(result[1]),
            ))

        return ImageLoader.loads(manifest)


class DBSource(Source):
    def __init__(
            self,
            db_hostname: str,
            db_username: str,
            db_password: str,
            db_name: str,
            base_url: str
    ):
        """
        Database Source Implementation
        Args:
            db_hostname: The hostname of database server
            db_username: The username of database server
            db_password: The password of database user
            db_name: The name of database to use
            base_url: The base url prefix append before images path
        """
        super(DBSource, self).__init__()

        import pymysql
        import threading

        self.base_url = base_url
        self.download_tasks = []
        self.manifest = []
        self.downloading = False

        # Init the connection to database
        self.connection = pymysql.Connection(
            user=db_username,
            password=db_password,
            host=db_hostname,
            database=db_name
        )

        self.base_db_url = 'mysql://' + self.connection.get_host_info().split()[-1] + '/' + db_name + '/'

        # Init two task threads
        self.fetcher = threading.Thread(target=self.fetch)
        self.downloader = threading.Thread(target=self.download)

        self.fetcher.start()
        self.downloader.start()

        # Wait both threads exit
        while True:
            if self.fetcher.is_alive():
                continue
            elif self.download_tasks:
                continue
            else:
                self.downloading = False
                break

        self.data = ImageLoader.loads(self.manifest)
        self.length = len(self.data)

    def fetch(self) -> None:
        # Two cursors to perform different queries
        cursor_1 = self.connection.cursor()
        cursor_2 = self.connection.cursor()

        course = 'SELECT * from `mtrl_info`'  # Query all materials
        cursor_1.execute(course)

        materials = []

        while True:
            result = cursor_1.fetchone()
            if not result:
                break
            material_id = result[0]

            # Query all params belongs to current material
            course = 'SELECT * from `mtrl_para` where `mtrl_id` = %d' % material_id
            cursor_2.execute(course)
            params = cursor_2.fetchall()

            # Init a material param list and fill it
            material = [None, None, None, None, None]
            for param in params:
                material[
                    param[2]
                ] = (param[3] if param[2] in [1, 2] else (self.base_url + param[7]))

            urls = material[3:]
            material[0] = self.base_db_url + 'mtrl_info/' + str(material_id)
            material[3] = 'images/' + material[3].split('/')[-1]
            material[4] = 'images/' + material[4].split('/')[-1]

            # Push download task into the download tasks list
            self.download_tasks.append({'url': urls[0], 'path': 'images/'})
            self.download_tasks.append({'url': urls[1], 'path': 'images/'})

            # Push the material params list into the materials list
            materials.append(material)

        self.manifest = materials

    def download(self):
        import requests

        # Downloading signal
        self.downloading = True

        while self.downloading:
            if self.download_tasks:
                task = self.download_tasks.pop()
                url = task['url']
                path = task['path']

                response = requests.get(url)
                filename = url.split('/')[-1]

                with open(path + filename, 'wb') as file:
                    file.write(response.content)
            else:
                continue


class SavedSource(Source):
    def __init__(self, path: str):
        """
        Load data from a previously saved source.
        Args:
            path: The path to pickle file
        """
        super(SavedSource, self).__init__()

        self.path = path
        self.data = []
        self.fetch()
        self.length = len(self.data)

    def fetch(self):
        with open(self.path, 'rb') as bin_file:
            self.data = pickle.load(bin_file)
