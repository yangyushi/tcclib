import re
import os
import json

import numpy as np
import pandas as pd


INT_TYPES = (
    int, np.int, np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64
)

class FrameIter:
    """
    Iter frame-wise data in a text file, organised as

    Frame 1
    Frame 2
    ...

    For each frame, the content is

    Header   # one-line stating start of frame
    Comment  # many-lines arbitrary texts to be skipped
    Data     # many-lines data to be loaded as a numpy array

    A typical application for this is to parse the XYZ file.


    Attributes:
        numbers (list): the number of particles in each frame
        __f (io.FileIO): a FileIO instance obtained by `open()`
        __frame_cursors (list): the stream position of the start of each frame,
            the cursor is located at the start of the 2nd line of the frame.
            (the comment line)
        __kwargs (dict): the arguments to load a single frame using function\
            `numpy.loadtxt` or `pandas.read_csv`.
        __engine (str): choose the engine to load the result to numpy array\
            [pandas]: the data were obtained with `pandas.read_csv`;\
            [numpy]: the data were obtained with `numpy.loadtxt`.\
            (in 2021, the pandas engine were ~10x faster)
        __func (callable): the function to be called to obtain results
    """
    def __init__(self, filename, header_pattern, n_comment, engine='pandas', **kwargs):
        self.numbers = []
        self.__frame = 0
        self.__frame_cursors = []
        self.__header_pattern = header_pattern
        self.__n_comment = n_comment
        self.__filename = filename
        self.__kwargs = {}
        self.__f = open(filename, 'r')

        if engine.lower() in ['pandas', 'pd', 'p']:
            self.__engine = 'pandas'
        elif engine.lower() in ['numpy', 'np', 'n']:
            self.__engine = 'numpy'
        self.__parse()
        self.__set_load_parameters(**kwargs)
        self.ndim = self.__detect_dimension()

    def __next__(self):
        if self.__frame < len(self):
            self.__frame += 1
            return self[self.__frame - 1]
        else:
            self.__frame = 0
            raise StopIteration

    def __getitem__(self, i):
        """
        Args:
            i (int): the frame number

        Return:
            np.ndarray: the information of all particles in a frame, shape (n, dim)
        """
        if type(i) in INT_TYPES:
            if self.numbers[i] == 0:
                return np.empty((0, self.ndim))
            self.__f.seek(self.__frame_cursors[i])
            if self.__engine == 'pandas':
                result = self.__func(
                    self.__f, nrows=self.numbers[i], **self.__kwargs
                ).values
            elif self.__engine == 'numpy':
                result = self.__func(
                    self.__f, max_rows=self.numbers[i], **self.__kwargs
                )
            else:
                raise ValueError("Unknown engine name, select from [numpy] or [pandas]")
            if result.ndim == 1:  # for frames with just 1 particle
                return result[np.newaxis, :]
            else:
                return result
        elif type(i) == slice:
            result = []
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self)
            step = i.step if i.step else 1
            for frame in range(start, stop, step):
                result.append(self[frame])
            return result

    def __len__(self):
        """
        return the total number of frames
        """
        return len(self.numbers)

    def __iter__(self): return self

    def __set_load_parameters(self, **kwargs):
        """
        this method is used to handle the difference between the
            two engines.
        """
        if 'delimiter' not in kwargs:  # use space as default delimiter
            kwargs.update({'delimiter': ' '})
        if self.__engine == 'numpy':
            self.__func = np.loadtxt
            self.__kwargs = {
                key : val for key, val in kwargs.items()
                if key not in ['skiprows', 'max_rows']
            }
        elif self.__engine == 'pandas':
            self.__func = pd.read_csv
            self.__kwargs = {
                key : val for key, val in kwargs.items()
                if key not in ['skiprows', 'nrows']
            }
            self.__kwargs.update({'index_col': False, 'header': None})
        else:
            raise ValueError(
                "Unknown engine name, select from [numpy] or [pandas]"
            )
        self.__kwargs['skiprows'] = self.__n_comment  # skip the comment

    def __parse(self):
        self.__frame_cursors = []
        self.numbers = []
        self.__f.seek(0)
        line = self.__f.readline()
        numbers = 0
        while line:
            is_head = re.match(self.__header_pattern, line)
            if is_head:
                self.numbers.append(numbers - self.__n_comment)
                cursor = self.__f.tell()
                self.__frame_cursors.append(cursor)
                numbers = 0
            else:
                numbers += 1
            line = self.__f.readline()
        self.numbers.pop(0)
        self.numbers.append(numbers)  # add the last frame

    def close(self):
        self.__f.close()

    def __del__(self):
        self.__f.close()

    def __detect_dimension(self):
        for i, num in enumerate(self.numbers):
            if num > 0:
                self.__f.seek(self.__frame_cursors[i])
                if self.__engine == 'pandas':
                    result = self.__func(
                        self.__f, nrows=self.numbers[i], **self.__kwargs
                    ).values
                elif self.__engine == 'numpy':
                    result = self.__func(
                        self.__f, max_rows=self.numbers[i], **self.__kwargs
                    )
                else:
                    raise ValueError("Unknown engine name, select from [numpy] or [pandas]")
                return result.shape[1]
        return 0

    def to_json(self, filename=None):
        """
        Save the essential data in the calss.

        Args:
            filename (bool or str): if the filename is None, the data will\
                be returned as a dictionary. If filename is str, the data\
                will be write to the harddrive.

        Return:
            None or dict: the essential data to reconstruct the object.
        """
        data = {
            'numbers': self.numbers,
            'filename': self.__filename,
            'frame': self.__frame,
            'frame_cursors': self.__frame_cursors,
            'header_pattern': self.__header_pattern,
            'n_comment': self.__n_comment,
            'engine': self.__engine,
            'ndim': self.ndim,
            'kwargs': self.__kwargs,  # TODO: ensure elements are serialisable
        }
        if isinstance(filename, type(None)):
            return data
        elif isinstance(filename, str):
            with open(filename, 'w') as f:
                json.dump(data, f)

    @classmethod
    def from_json(cls, data):
        """
        Create a frame iterable without parsing the file.\
            Instead load the metadata from a dict, or a\
            json file on the disk.

        Args:
            data (dict or str): a dictionary containing all elements\
            (see `FrameIter.to_json`), or a string to the json file\
            containing the dict.

        Example:
            >>> obj = FrameIter('sample.xyz')

            >>> cache = obj.to_json()  # save data in memory
            >>> new_obj = FrameIter.from_json(cache)

            >>> obj.to_json("cache.json")  # save data in disk
            >>> new_obj = FrameIter.from_json("cache.json")
        """
        if isinstance(data, str):
            with open(data, 'r') as f:
                data = json.load(f)
        elif isinstance(data, dict):
            pass
        else:
            raise TypeError(
                "Invalid datatype"
            )
        self = cls.__new__(cls)  # bypass __init__
        self.filename = data['filename']
        self.numbers = data['numbers']
        self.__frame = data['frame']
        self.__frame_cursors = data['frame_cursors']
        self.__header_pattern = data['header_pattern']
        self.__n_comment = data['n_comment']
        self.__engine = data['engine']
        if self.__engine == 'numpy':
            self.__func = np.loadtxt
        elif self.__engine == 'pandas':
            self.__func = pd.read_csv
        else:
            raise ValueError(
                "Unknown engine name, select from [numpy] or [pandas]"
            )
        self.ndim = data['ndim']
        self.__kwargs = data['kwargs']
        self.__f = open(self.filename)
        return self


class XYZ(FrameIter):
    """
    Fast XYZ parser that can handle very large xyz file

    """
    def __init__(self, filename, engine='pandas', align_opt=False, **kwargs):
        """
        Args:
            filename (str): the path to the xyz file to be loaded.
            engine (str): choose between pandas or numpy, pandas is faster
            align_opt (bool): Significantly (!) optimise the parsing speed\
                if the data in the xyz file is *right-aligned*, meaning\
                all coordinates have the same column width. If the \
                optimisation was mistakenly used for not aligned data,\
                an runtime error will be raised.
        """
        if align_opt: self._FrameIter__parse = self.__fast_parse
        super().__init__(
            filename,
            header_pattern=r'(\d+)\n',
            n_comment=1,
            engine=engine,
            **kwargs,
        )

    def __detect_line_offset(self):
        """
        Find the byte offset one one line in the data.

        Exampe:
            >>> lines_to_jump = 1000
            >>> offset = self.__detect_line_offset()
            >>> new_location = self.__f.tell() + offset * 1000
            >>> self.__f.seek(new_location)
        """

        f = self._FrameIter__f
        hp = self._FrameIter__header_pattern
        nc = self._FrameIter__n_comment
        f.seek(0)

        line = f.readline()
        while line:
            is_head = re.match(hp, line)
            if is_head:
                for _ in range(nc):
                    f.readline()
                cursor_before_line = f.tell()
                line = f.readline()
                if not re.match(hp, line):
                    return f.tell() - cursor_before_line
                else:
                    line = f.readline()
        raise RuntimeError("Can't detect the line offset")

    def __fast_parse(self):
        lo = self.__detect_line_offset()
        self._FrameIter__frame_cursors = []
        fcs = self._FrameIter__frame_cursors
        f = self._FrameIter__f
        hp = self._FrameIter__header_pattern
        nc = self._FrameIter__n_comment

        self.numbers = []
        f.seek(0)
        line = f.readline()

        while line:
            is_head = re.match(hp, line)
            if is_head:
                cursor = f.tell()
                fcs.append(cursor)
                n_particle = int(re.match('(\d+)\n', line).group(0))
                self.numbers.append(n_particle)
                for _ in range(nc):
                    f.readline()
                f.seek(f.tell() + n_particle * lo)
                line = f.readline()
            else:
                raise RuntimeError(
                    "Failed to parse the xyz file with align optimisation"
                )


class ClusterOutput(FrameIter):
    """
    The parser for the cluster file generated by TCC, like the XYZ class,\
        it can handle very large file (long trajectories of many particles).
    """
    def __init__(self, filename, engine='pandas', **kwargs):
        super().__init__(
            filename,
            header_pattern='Frame Number \d+\n',
            n_comment=0,
            engine=engine,
            **kwargs
        )


def dump_xyz(filename, positions, comment='', precision=10):
    """
    Dump positions into an xyz file. The data columns will be right aligned,
        enabling optimised parsing by `XYZ`.

    Args:
        filename (str): the name of the xyz file, it can be an existing file
        positions (numpy.ndarray): the positions of particles, shape (n, dim)
        comment (str): the content in the comment line

    Return:
        None
    """
    n, dim = positions.shape
    fmt = f'%.{precision}e'
    with open(filename, 'a') as f:
        np.savetxt(
            f, positions, delimiter=' ',
            header='%s\n%s' % (n, comment),
            comments='',
            fmt=['A ' + fmt] + [fmt for i in range(dim - 1)]
        )


def make_movie_bool(xyz, tcc, output, start=0, stop=None, clusters=None):
    if output in os.listdir('.'):
        os.remove(output)
    if isinstance(clusters, type(None)):
        clusters = tcc.cluster_bool.keys()
    if isinstance(stop, type(None)):
        stop = len(tcc)
    comment = "x y z " + " ".join(clusters)
    for f in range(start, stop):
        coord = xyz[f]  # n, 3
        is_cluster = tcc.frame_bool(f, clusters).to_numpy()  # n, m
        data = np.concatenate((coord, is_cluster), axis=1)  # (n, m + 3)
        dump_xyz(output, data, comment=comment)


def make_movie_count(xyz, tcc, output, start=0, stop=None, clusters=None):
    if output in os.listdir('.'):
        os.remove(output)
    if isinstance(clusters, type(None)):
        clusters = tcc.cluster_bool.keys()
    if isinstance(stop, type(None)):
        stop = len(tcc)
    comment = "x y z " + " ".join(clusters)
    for f in range(start, stop):
        coord = xyz[f]  # n, 3
        is_cluster = tcc.frame_count(f, clusters).to_numpy()  # n, m
        data = np.concatenate((coord, is_cluster), axis=1)  # (n, m + 3)
        dump_xyz(output, data, comment=comment)
