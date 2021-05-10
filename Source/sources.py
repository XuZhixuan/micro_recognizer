import os
import pickle
import re
import zipfile
from abc import ABC
from typing import List, Tuple

from Tools import ImageLoader
from container import Container


class Source(ABC):
    _chunk = -1
    _chunks = []
    _dumping = True
    _chunk_size = 1000

    def __init__(self, app: Container, chunk_size: int = 1000):
        self._chunk_size = chunk_size
        self.data = []
        self.length = 0
        self.app = app
        self.loader = app.resolve(ImageLoader)

    def __iter__(self):
        self._current = 0
        return self

    def __next__(self):
        index = self._current
        self._current += 1
        if index >= self.length:
            raise StopIteration
        return self[index]

    def __getitem__(self, item):
        if isinstance(item, slice):
            if isinstance(self, SavedSource):
                chunks = len(self._chunks)
                start = 0 if item.start is None else item.start
                stop = 0 if item.stop is None else item.stop
                _start = int(start / self.length * chunks)
                _stop = int((stop - start) / self.length * chunks) + start
                new_chunks = self._chunks[_start:_stop]
                source = SavedSource(self.app, self.path)
                source._chunks = new_chunks
                source.length = sum(map(lambda x: x[1], new_chunks))
                return source

            full_list = []
            for chunk in self._chunks:
                full_list.extend(chunk)
            new_list = full_list[item.start:item.stop]
            source = Source(self.app)
            source._chunks = list(self.app.helper.list_chunk(new_list, self._chunk_size))
            source.length = len(new_list)
            return source

        if self._chunk == -1 or self._chunk != int(item / self._chunk_size):
            self._chunk = int(item / self._chunk_size)
            self.data = self.loader(
                self._chunks[self._chunk]
            )

        image = self.data[item % self._chunk_size]

        return image.grayscale, image.thermal

    def __len__(self):
        return self.length

    def dump(self, path: str):
        with open(path, 'wb') as bin_file:
            pickle.dump(self.data, bin_file)

    def load(self, path: str):
        with open(path, 'rb') as bin_file:
            self.data = pickle.load(bin_file)

    def chunks_dump(self):
        name = self.app.helper.time_name()
        os.mkdir(self.app.config('data.cache') + name)

        meta = []
        for i, datum in enumerate(self._chunks):
            self.data = self.loader(datum)
            self.dump(self.app.config('data.cache') + name + '/' + str(i) + '.pkl')
            meta.append((str(i) + '.pkl', len(self.data)))

        self.app.helper.dump_json(self.app.config('data.cache') + name + '/meta.json', meta)


class FileSource(Source):
    def __init__(self, app: Container, dir_name: str, dump: bool = True):
        super().__init__(app)
        self.dir_name = dir_name
        self._dumping = dump
        self.fetch()

    def fetch(self):
        # Detect the zip files presented in given dir
        zip_files = []
        for file in os.listdir(self.dir_name):
            if re.match(r'(.*)\.zip', file):
                zip_files.append(
                    zipfile.ZipFile(self.dir_name + '/' + file)
                )

        manifest = []
        # Load data from each zip file
        for zip_file in zip_files:
            manifest += self.load_list(zip_file)
            self.length += len(manifest)

        # 加载列表分块
        self._chunks = list(self.app.helper.list_chunk(manifest, self._chunk_size))

        if self._dumping:
            self.chunks_dump()

    @staticmethod
    def load_list(file: zipfile.ZipFile) -> list:
        nums = []
        # Determine the first file in zip file
        for name in file.namelist():
            # Extract every file from zip file
            file.extract(name, path='./storage/images')
            num = re.search(r'GS(.*)\.png', name)
            if num:
                nums.append(
                    int(num.group(1))
                )

        start = min(nums)
        # Determine the name of extract dir
        dir_name = file.namelist()[0].split('/')[0]

        # Transform the result file into a load list
        with open('./storage/images/' + dir_name + '/result.txt', 'r') as results_file:
            results = results_file.readlines()

        manifest = []
        for i, result in enumerate(results, start):
            result = result.split()
            manifest.append((
                file.filename + ':GS' + str(i) + '.png',
                './storage/images/' + dir_name + '/GS' + str(i) + '.png',
                './storage/images/' + dir_name + '/RGB' + str(i) + '.png',
                float(result[0]),
                float(result[1]),
            ))

        return manifest


class DBSource(Source):
    _chunk = -1
    _chunks = []
    _dumping = True

    def __init__(
            self,
            app: Container,
            db_hostname: str,
            db_username: str,
            db_password: str,
            db_name: str,
            base_url: str,
            dump: bool = True
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
        super(DBSource, self).__init__(app)

        import pymysql
        import threading

        self._dumping = dump
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

        self._chunks = list(self.app.helper.list_chunk(self.manifest, self._chunk_size))
        self.length = len(self.manifest)

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

        self._chunks = list(self.app.helper.list_chunk(materials, self._chunk_size))
        self.length = len(materials)

        if self._dumping:
            self.chunks_dump()

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
    def __init__(self, app: Container, dir_name: str):
        """
        Load data from a previously bin source.
        Args:
            dir_name: The path to saves file
        """
        super(SavedSource, self).__init__(app)
        self.path = dir_name
        self.fetch()
        self.loader = self.load_chunk

    def fetch(self):
        meta = self.app.helper.load_json(self.path + '/meta.json')
        for chunk_desc in meta:
            self._chunks.append(tuple(chunk_desc))
            self.length += chunk_desc[1]

    def load_chunk(self, chunk_desc):
        with open(self.path + '/' + chunk_desc[0], 'rb') as bin_file:
            return pickle.load(bin_file)


class TestSource(Source):
    resources = [
        ('tests/train-images-idx3-ubyte', 'tests/train-labels-idx1-ubyte'),
        ('tests/t10k-images-idx3-ubyte', 'tests/t10k-labels-idx1-ubyte')
    ]

    def __init__(self, app: Container):
        super(TestSource, self).__init__(app)
        self.data = self.fetch()
        pass

    def fetch(self) -> List[Tuple]:
        import struct
        from PIL import Image

        data = []

        for images, labels in self.resources:
            with open(images, 'br') as img_files, open(labels, 'br') as lab_files:
                # Read the magic number
                img_files.read(4)
                lab_files.read(8)
                # Read the file number
                num = img_files.read(4)
                num = struct.unpack('>i', num)[0]
                # Read the size of image
                height = struct.unpack('>i', img_files.read(4))[0]
                width = struct.unpack('>i', img_files.read(4))[0]
                size = width * height

                padding = b'\x00\x00\x00'
                # Read the images
                for i in range(num):
                    img = img_files.read(size)
                    lab = lab_files.read(1)
                    # Convert format
                    label = struct.unpack('>i', padding + lab)[0]
                    image = Image.frombytes(
                        mode='L',
                        size=(28, 28),
                        data=img
                    )

                    data.append((
                        self.loader.loader(image).unsqueeze(0).cuda(),
                        label
                    ))

        return data
