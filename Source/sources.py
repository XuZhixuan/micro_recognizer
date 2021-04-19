from abc import abstractmethod

import pickle

from Tools import ImageLoader


class Source:
    @abstractmethod
    def fetch(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def dump(self, path: str):
        raise NotImplementedError


class FileSource(Source):
    def __init__(self):
        pass

    def fetch(self):
        pass

    def __iter__(self):
        pass

    def __getitem__(self, item):
        pass

    def __next__(self):
        pass

    def dump(self, path: str):
        pass


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
        pass

    def fetch(self) -> None:
        import re

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
            material[0] = re.search(r'/RGB(.*).png', material[4]).group(1)

            # Push download task into the download tasks list
            self.download_tasks.append({'url': material[3], 'path': 'images/'})
            self.download_tasks.append({'url': material[4], 'path': 'images/'})

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

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr >= self.length:
            raise StopIteration
        else:
            data = self.data[self.curr]
            self.curr += 1
            return data

    def __getitem__(self, item):
        return self.data[item]

    def dump(self, path: str):
        with open(path, 'wb') as bin_file:
            pickle.dump(self.data, bin_file)


class SavedSource(Source):
    def __init__(self, path: str):
        self.path = path
        self.data = []
        self.fetch()
        self.length = len(self.data)

    def fetch(self):
        with open(self.path, 'rb') as bin_file:
            self.data = pickle.load(bin_file)

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr >= self.length:
            raise StopIteration
        else:
            data = self.data[self.curr]
            self.curr += 1
            return data

    def __getitem__(self, item):
        return self.data[item]

    def dump(self, path: str):
        with open(path, 'wb') as bin_file:
            pickle.dump(self.data, bin_file)
