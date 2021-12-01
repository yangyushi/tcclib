import re
import os
from glob import glob
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import json
import configparser

from .utility import dump_xyz


INT_TYPES = (
    int, np.int, np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64
)
ROOT = os.path.split(__file__)[0]
TCC_EXEC = os.path.abspath(f"{ROOT}/tcc")


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
    def __init__(self, filename, engine='pandas', **kwargs):
        super().__init__(
            filename,
            header_pattern=r'(\d+)\n',
            n_comment=1,
            engine=engine,
            **kwargs,
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


class TCC:
    """
    A light-weight python wrapper for Topological Cluster Classification.
    It is especially designed to handle very large xyz files.
    """
    def __init__(self, work_dir="", load_cache=True):
        """
        Args:
            work_dir (str): the path to the folder containing all tcc output\
                files. If the folder does not exist, it will be created when\
                `TCC.run` is called.
            load_cache (bool): if a cached file exists in the work_dir, then\
                use the cache to avoid repeated parsing.
        """
        if not work_dir:
            work_dir = "."
        self.__dir = work_dir
        self.__raw_dir = os.path.join(self.__dir, 'raw_output')
        self.__cluster_dir = os.path.join(self.__dir, 'cluster_output')
        cache_fn = os.path.join(self.__dir, 'pyTCC_cache.json')
        if load_cache and os.path.isfile(cache_fn):
            with open(cache_fn, 'rb') as f:
                cache = json.load(f)
            self.clusters_to_analyse = cache['clusters_to_analyse']
            self.cluster_bool = {
                name: XYZ.from_json(data_obj)
                for name, data_obj in cache['cluster_bool'].items()
            }
            self.cluster_detail = {
                name: XYZ.from_json(data_obj)
                for name, data_obj in cache['cluster_detail'].items()
            }
        else:
            self.clusters_to_analyse = []
            self.cluster_bool = {}    # if a particle is in different clusters
            self.cluster_detail = {}  # the particle indices of each cluster

    def __len__(self):
        """
        The frame number
        """
        if self.cluster_bool:  # not an empty cluster
            for key in self.cluster_bool:
                return len(self.cluster_bool[key])
        else:
            return 0

    def __write_box(self, box):
        """
        Generate a legal box.txt file in the tcc working directory
        """
        if box.ndim == 1:
            box = box[np.newaxis, :]  # (3,) --> (1, 3)
        if box.shape[1] != 3:
            raise RuntimeError("The dimension of box should be 3")
        box = np.concatenate((
            np.arange(box.shape[0])[:, np.newaxis],  # shape (n, 1)
            box  # shape (n, 3)
        ), axis=1)  # shape (n, 4)
        np.savetxt(
            fname=os.path.join(self.__dir, 'box.txt'),
            X=box,
            header='#iter Lx Ly Lz', fmt=["%d"] + ["%.8f"] * 3
        )

    def __write_parameters(self, **kwargs):
        """
        Write inputparameters.ini and clusters_to_analyse.ini

        A symlink named `sample` should be inside self__dir, linking
            to the xyz file to be analysed
        """
        input_parameters = {
            "Box": { "box_type": 1, "box_name": "box.txt" },
            "Run": { "xyzfilename": "sample", "frames": 1},
            "Simulation": {
                "rcutAA": 1.8, "rcutAB": 1.8, "rcutBB": 1.8, "min_cutAA": 0.0,
                "bond_type": 1, "PBCs": 1, "voronoi_parameter": 0.82,
                "num_bonds": 50, "cell_list": 1, "analyse_all_clusters": 1,
            },
            "Output": {
                "bonds": 0, "clusts": 0, "raw": 1, "do_XYZ": 0,
                "11a": 0, "13a": 0, "pop_per_frame": 1,
            }
        }

        clusters = { "Clusters": {
            "sp3a": 0, "sp3b": 0, "sp3c": 0, "sp4a": 0, "sp4b": 0, "sp4c": 0,
            "sp5a": 0, "sp5b": 0, "sp5c": 0, "6A": 0, "6Z": 0, "7K": 0,
            "7T_a": 0, "7T_s": 0, "8A": 0, "8B": 0, "8K": 0, "9A": 0, "9B": 0,
            "9K": 0, "10A": 0, "10B": 0, "10K": 0, "10W": 0, "11A": 0,
            "11B": 0, "11C": 0, "11E": 0, "11F": 0, "11W": 0, "12A": 0,
            "12B": 0, "12D": 0, "12E": 0, "12K": 0, "13A": 0, "13B": 0,
            "13K": 0, "FCC": 0, "HCP": 0, "BCC_9": 0, "BCC_15": 0,
            }
        }

        for key in kwargs:
            for section in input_parameters:
                if key in input_parameters[section].keys():
                    input_parameters[section][key] = kwargs[key]

        for key in clusters["Clusters"]:
            if key in self.clusters_to_analyse:
                clusters["Clusters"][key] = 1

        config_input = configparser.ConfigParser()
        config_input.read_dict(input_parameters)
        config_cluster = configparser.ConfigParser()
        config_cluster.read_dict(clusters)

        with open(
            os.path.join(self.__dir, "inputparameters.ini"), 'w'
        ) as f:
            config_input.write(f)
        with open(
            os.path.join(self.__dir, "clusters_to_analyse.ini"), 'w'
        ) as f:
            config_cluster.write(f)

    def run(self, xyz, box, frames=None, tcc_exec="", silent=True, **kwargs):
        """
        Call tcc to analyse an XYZ file. The output will be write to `self.__dir`

        Args:
            xyz (str): the path to the xyz file to be analysed. Notice that
                the working directory is `self.__dir` if using a relative path.
            box (list): the box of the simulation / experiment. The warpper
                supports different boxes in different frames.
            frames (int): the number of frames to perform TCC analysis. This
                wrapper DOES NOT check if this value is valid.
            tcc_exec (str): the path to TCC binary executable, if it is empty\
                the default tcc from extern/TCC will be used.
            silent (bool): if True the output of TCC will be supressed
            kwargs (dict): tcc parameters. These parameters will overwrite\
                the default parameters.

        Return:
            None
        """
        if not os.path.isfile(xyz):
            raise FileNotFoundError("The xyz file does not exist: ", xyz)
        if os.path.isabs(xyz):
            xyz_path = xyz
        else:
            xyz_path = os.path.abspath(xyz)

        self.xyz = XYZ(xyz_path)
        if not tcc_exec:
            tcc_exec = TCC_EXEC  # use the default TCC executable

        if (self.__dir not in os.listdir(os.getcwd())) and (self.__dir != "."):
            os.makedirs(os.path.join(os.getcwd(), self.__dir))

        self.__write_box(np.array(box))

        # create a soft link of the xyz file to self.__dir
        soft_link = os.path.join(self.__dir, "sample")
        if os.path.isfile(soft_link):
            os.remove(soft_link)
        os.symlink(src=xyz_path, dst=soft_link)

        if isinstance(frames, type(None)):
            frames = len(XYZ(xyz_path))
        self.__write_parameters(frames=frames, **kwargs)
        if silent:
            subprocess.run(
                args=tcc_exec,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.__dir,
                check=True,
            )
        else:
            subprocess.run(args=tcc_exec, cwd=self.__dir, check=True)

    def __parse_raw(self):
        if not os.path.isdir(self.__raw_dir):
            raise FileNotFoundError(
                "No raw_output folder, rerun the tcc with Output.raw = 1"
            )
        cluster_name_pattern = re.compile(r'sample.*raw_(.+)')
        filenames = glob(
            "{folder}/sample*raw_*".format(folder=self.__raw_dir)
        )
        filenames = [os.path.basename(fn) for fn in filenames]
        filenames.sort()
        cluster_names = [
            cluster_name_pattern.match(fn).group(1) for fn in filenames
        ]
        for cn in cluster_names:
            fn = glob(
                "{folder}/sample*raw_{cluster_name}".format(
                    folder=self.__raw_dir, cluster_name=cn
                )
            )
            if len(fn) == 0:
                raise FileNotFoundError(
                    "Raw output file for {c} not found".format(c=cn)
                )
            if len(fn) > 1:
                raise RuntimeError(
                    "Multiple raw file found for {c}".format(c=cn)
                )
            else:
                fn = fn[0]
            self.cluster_bool.update({
                cn: XYZ(fn, true_values=['C'], false_values=['A'])
            })

    def __parse_cluster(self):
        if not os.path.isdir(self.__cluster_dir):
            raise FileNotFoundError(
                "No cluster_output folder, rerun the tcc with Output.cluster = 1"
            )
        cluster_name_pattern = re.compile(r'sample.*clusts_(.+)')
        filenames = glob(
            "{folder}/sample*clusts_*".format(folder=self.__cluster_dir)
        )
        filenames = [os.path.basename(fn) for fn in filenames]
        filenames.sort()
        cluster_names = [
            cluster_name_pattern.match(fn).group(1) for fn in filenames
        ]
        for cn in cluster_names:
            fn = glob(
                "{folder}/sample*clusts_{cluster_name}".format(
                    folder=self.__cluster_dir, cluster_name=cn
                )
            )
            if len(fn) == 0:
                raise FileNotFoundError(
                    "Raw output file for {c} not found".format(c=cn)
                )
            if len(fn) > 1:
                raise RuntimeError(
                    "Multiple raw file found for {c}".format(c=cn)
                )
            else:
                fn = fn[0]
            self.cluster_detail.update({
                cn: ClusterOutput(fn, delimiter='\t')
            })

    def parse(self, raw=True, cluster=True, cache=True):
        self.__parse_raw()
        self.__parse_cluster()
        if cache:
            self.__cache()

    def frame_bool(self, f, clusters=None):
        """
        Getting the result of particles and the clusters they are in. The
            output is a numpy array of Boolean values. One example would be

        ..code-block::

            id, FCC, 13A, 12E, 11F, 10B
            1,    0,   0,   0,   0,   1   # particle 1 is in 10B
            2,    1,   0,   0,   0,   0   # particle 2 is in FCC
            3,    0,   0,   0,   0,   0   # particle 3 is not in any cluster
            ...

        Args:
            f (int): the frame number
        """
        if isinstance(clusters, type(None)):
            clusters = self.cluster_bool.keys()
        result_dict = {}
        for cn in clusters:
            result_dict[cn] = self.cluster_bool[cn][f].ravel()
        return pd.DataFrame.from_dict(data=result_dict, orient='columns')

    def frame_count(self, f, clusters=None):
        """
        Getting the result of particles and the clusters they are in. The
            output is a numpy array of Boolean values. One example would be

        ..code-block::

            id, FCC, 13A, 10B, ...
            1,    2,   0,   0, ... # particle 1 is in 2 10B, 0 13A, and 0 10B
            2,    0,   1,   3, ... # particle 2 is in 0 FCC, 1 13A, and 3 10B
            ...

        Args:
            f (int): the frame number
        """
        if isinstance(clusters, type(None)):
            clusters = self.cluster_detail.keys()
        result_dict = {}

        for cn in clusters:
            n = self.cluster_bool[cn][f].shape[0]
            count = np.zeros(n, dtype=int)
            for cluster_indices in self.cluster_detail[cn][f]:
                count[cluster_indices] += 1
            result_dict[cn] = count

        return pd.DataFrame.from_dict(data=result_dict, orient='columns')

    @property
    def population(self):
        """
        Return the mean population if each frame as a pandas table

        Return:
            pandas.DataFrame: a pandas table where each columns are the clusters
                and each rows are different frames
        """
        pattern = 'sample*.pop_per_frame'
        fn = glob(os.path.join(self.__dir, pattern))
        if len(fn) == 1:
            fn = fn[0]
        elif len(fn) == 0:
            raise FileNotFoundError(
                "Population output file ({p}) not found".format(p=pattern)
            )
        else:
            raise RuntimeError("Multiple population output files exist")
        df = pd.read_csv(fn, sep='\t', header=0, index_col=0)
        df.dropna(axis='columns', inplace=True)
        return df.T

    def __cache(self):
        """
        Save the essential data to the hard disk, avoiding repeated parsing.
        """
        cache = {
            'cluster_bool': {
                name: data_obj.to_json()
                for name, data_obj in self.cluster_bool.items()
            },
            'cluster_detail': {
                name: data_obj.to_json()
                for name, data_obj in self.cluster_detail.items()
            },
            'clusters_to_analyse': self.clusters_to_analyse,
        }
        cache_fn = os.path.join(self.__dir, 'pyTCC_cache.json')
        with open(cache_fn, 'w') as f:
            json.dump(cache, f)


class TCCOTF(TCC):
    """
    A light-weight python wrapper for TCC. The calculation is "on the fly", where
    """
    def __init__(self):
        self.__tmp_dir = TemporaryDirectory()
        TCC.__init__(self, self.__tmp_dir.name)


    def run(self, configurations, box, tcc_exec="", silent=True, **kwargs):
        """
        Call tcc to analyse an XYZ file. The coordinates will be write to the hard disk
            temporarily on a random location, and the temp directory will be removed
            upon the destruction of the TCCOTF object.

        Args:
            configurations (numpy.ndarray): the particle coordinaes in\
                different time points. The shape of the array should be\
                (n_frame, n_particle, 3)
            box (iterable): the box of the simulation / experiment. This\
                warpper supports different boxes in different frames.
            tcc_exec (str): the path to TCC binary executable.
            silent (bool): if True the output of TCC will be supressed
            kwargs (dict): tcc parameters. These parameters will overwrite\
                the default parameters.

        Return:
            None
        """
        if not tcc_exec:
            tcc_exec = TCC_EXEC  # use the default TCC executable

        xyz_name = os.path.join(self._TCC__dir, 'sample')
        for i, conf in enumerate(configurations):
            dump_xyz(xyz_name, conf, comment=i+1)

        self._TCC__write_box(np.array(box))
        self._TCC__write_parameters(frames=len(configurations), **kwargs)

        print(self._TCC__dir)

        if silent:
            subprocess.run(
                args=tcc_exec,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self._TCC__dir,
                check=True,
            )
        else:
            subprocess.run(args=tcc_exec, cwd=self._TCC__dir, check=True)
