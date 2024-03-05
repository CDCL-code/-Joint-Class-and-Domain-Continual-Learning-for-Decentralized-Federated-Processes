"""Adapted from: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/dassl/data/datasets/da/domainnet.py """

#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

import os.path as osp
from collections import defaultdict
from typing import List, Sequence
from avalanche.benchmarks.datasets.downloadable_dataset import DownloadableDataset
from torchvision.datasets.folder import default_loader
from avalanche.benchmarks import NCScenario, nc_benchmark, ni_benchmark, nc_nd_benchmark, nc_nd_benchmarkx2


def get_key_from_value(d, val):
        keys = [k for k, v in d.items() if v == val]
        if keys:
            return keys[0]
        return None

class Datum:
    """Data instance which defines the basic attributes.
    Args:
        impath (str): image path.
        label (int): class label.
        domain (int): domain label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, domain=0, classname=""):
        assert isinstance(impath, str)

        self._impath = impath
        self._label = label
        self._domain = domain
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def domain(self):
        return self._domain

    @property
    def classname(self):
        return self._classname


class DomainNet(DownloadableDataset):
    """DomainNet.
    Based on: https://github.com/KaiyangZhou/Dassl.pytorch

    How to install: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#domainnet
    Download link: http://ai.bu.edu/M3SDA/. (Please download the cleaned version of split files)
    Splitfile for Mini-DomainNet: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md#minidomainnet
    File structure:

    domainnet/
        |–– clipart/
        |–– infograph/
        |–– painting/
        |–– quickdraw/
        |–– real/
        |–– sketch/
        |–– splits/
        |   |–– clipart_train.txt
        |   |–– clipart_test.txt
        |   |–– ...
        |–– splits_mini/ (OPTIONAL FOR MINI-SPLITS)
        |   |–– clipart_train.txt

    Statistics:
        - 6 distinct domains: Clipart, Infograph, Painting, Quickdraw,
        Real, Sketch.
        - Around 0.6M images.
        - 345 categories.
        - URL: http://ai.bu.edu/M3SDA/.
    Special note: the t-shirt class (327) is missing in painting_train.txt.
    Reference:
        - Peng et al. Moment Matching for Multi-Source Domain
        Adaptation. ICCV 2019.
    """

    dataset_dir = "domainnet"
    domains = [
        "clipart", "infograph", "painting", "quickdraw", "real", "sketch"
    ]

    # (Dataset,train_split,test_split)
    domain_to_urls = {
        "clipart": ("http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt",
                    "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt"),
        "infograph": ("http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
                      "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_train.txt",
                      "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/infograph_test.txt"),
        "painting": ("http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
                     "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_train.txt",
                     "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/painting_test.txt"),
        "quickdraw": ("http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
                      "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_train.txt",
                      "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/quickdraw_test.txt"),
        "real": ("http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
                 "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_train.txt",
                 "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/real_test.txt"),
        "sketch": ("http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip",
                   "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt",
                   "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt"),
    }

    def __init__(self, ds_root, domain: str, train: bool = True,
                 transform=None, target_transform=None, loader=default_loader, download=True,
                 ):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.domain = domain
        assert domain in self.domains
        assert domain in self.domain_to_urls

        # Paths
        self.dataset_dir = osp.join(ds_root, self.dataset_dir)
        super().__init__(root=self.dataset_dir, download=download, verbose=True)
        self.split_dir = self.dataset_dir  # Same dir as main
        """ root/domainnet"""

        # Download
        self.url_data = self.domain_to_urls[self.domain][0]  # Download link
        self.url_trainsplit = self.domain_to_urls[self.domain][1]  # Download link
        self.url_testsplit = self.domain_to_urls[self.domain][2]  # Download link
        self._load_dataset()

        # Preprocessing: Read Original dataset
        self.split = "train" if self.train else "test"
        self.data: list = self._read_data([self.domain], split=self.split)
        self.set_meta_data()

    @property
    def targets(self):
        """ For compatibility with AvalancheDataset."""
        targets = []
        for item in self.data:
            targets.append(item.label)
        return targets

    def set_meta_data(self):
        self._num_classes = self.get_num_classes(self.data)
        self._lab2cname, self._classnames = self.get_lab2cname(self.data)

    def _read_data(self, input_domains: list, split="train"):
        if isinstance(input_domains, str):
            input_domains = [input_domains]

        items = []
        for domain, dname in enumerate(input_domains):
            filename = dname + "_" + split + ".txt"
            split_file = osp.join(self.split_dir, filename)

            with open(split_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    impath, label = line.split(" ")
                    classname = impath.split("/")[1]
                    impath = osp.join(self.dataset_dir, impath)
                    label = int(label)
                    item = Datum(
                        impath=impath,
                        label=label,
                        domain=domain,
                        classname=classname
                    )
                    items.append(item)
        return items

    def get_num_classes(self, data_source):
        """Count number of classes.
        Args:
            data_source (list): a list of Datum objects.
        """
        label_set = set()
        for item in data_source:
            label_set.add(item.label)
        return len(label_set)
    
    


    def get_lab2cname(self, data_source):
        """Get a label-to-classname mapping (dict).
        Args:
            data_source (list): a list of Datum objects.
        """
        container = set()
        for item in data_source:
            container.add((item.label, item.classname))
        mapping = {label: classname for label, classname in container}
        labels = list(mapping.keys())
        labels.sort()
        classnames = [mapping[label] for label in labels]
        return mapping, classnames

    def _download_error_message(self) -> str:
        return 'Error downloading the dataset. Consider downloading ' \
               'it manually at: ' + self.url_data + ' and placing it ' \
                                                    'in: ' + str(self.root)

    def _download_dataset(self) -> None:
        """
        The download procedure.

        This procedure is called only if `_load_metadata` fails.

        This method must raise an error if the dataset can't be downloaded.

        Hints: don't re-invent the wheel! There are ready-to-use helper methods
        like `_download_and_extract_archive`, `_download_file` and
        `_extract_archive` that can be used.

        :return: None
        """
        print(f"Downloading dataset on {self.url_data}")
        zip_filename = osp.basename(self.url_data)  # How to store zip-file

        # Download the dataset, downloaded to the root given to super
        finished_token_file = f"{zip_filename.split('.')[0]}.DOWNLOADED_EXTRACTED"
        finished_token_path = osp.join(self.root, finished_token_file)

        if osp.isfile(finished_token_path):
            print(f"Skipping downloading/extraction of dataset, processing token exists: {finished_token_path}")
        else:
            self._download_and_extract_archive(self.url_data, zip_filename, None,
                                               sub_directory=None,  # Already has sub_dir with name
                                               remove_archive=True  # True
                                               )
            # Save finished token (empty file)
            open(finished_token_path, 'a').close()
            print(f"Saved processing token: {finished_token_path}")

    def _load_metadata(self) -> bool:
        """
        The dataset metadata loading procedure.

        This procedure is called at least once to load the dataset metadata.

        This procedure should return False if the dataset is corrupted or if it
        can't be loaded.

        :return: True if the dataset is not corrupted and could be successfully
        loaded.
        """
        # Downlaod both metadata and the
        self._download_file(self.url_trainsplit, file_name=osp.basename(self.url_trainsplit), checksum=None)
        self._download_file(self.url_testsplit, file_name=osp.basename(self.url_testsplit), checksum=None)
        self._download_dataset()
        return True

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target
            class.
        """
        fpath = self.data[index].impath
        target = self.data[index].label

        sample = self.loader(str(self.root / fpath))

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Domain: {}\n'.format(self.domain)
        fmt_str += '    Train: {}\n'.format(self.train)
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace(
                '\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace(
                '\n', '\n' + ' ' * len(tmp)))
        return fmt_str


    
class MiniDomainNet(DomainNet):
    """
    Subset of less noisy labels from DomainNet.
    Only 4 of the 6 domains are considered. Class-split selected by
    https://arxiv.org/pdf/1904.06487.pdf and https://arxiv.org/pdf/2003.07325.pdf

    For MiniDomainNet splits (4 domains), see: https://drive.google.com/open?id=15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7)
    """
    nb_classes = 10
    domains = ['clipart', 'real']  # No infograph,'clipart', 'painting', 'real', 'quickdraw', 'sketch'
    classes_list =  [ 77, 100,  117, 256,270, 282, 291, 293, 294,  344]
    assert len(classes_list) == nb_classes
    """ Subset of less noisy labels, class-split selected by 
    https://arxiv.org/pdf/1904.06487.pdf and https://arxiv.org/pdf/2003.07325.pdf"""

    def __init__(self, classes_list: List[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Loads all data

        if classes_list is None:
            classes_list = self.classes_list

        # Keep original labels for summaries, but return adapted (in [0,N] range) label with adapter
        self.orig_to_new_target_map = {old_target: new_target
                                       for new_target, old_target in enumerate(sorted(classes_list))}
        self.targets_adapter = lambda x: int(self.orig_to_new_target_map[int(x)])
        """targets_adapter: Function mapping target to other target. (e.g. transform label-subset to valid
                                [0,N] range). Mapped at __get_item__ and calling the .target property."""

        # Subset the data
        self.data = self.subset_only_with_labels(self.data, classes_list)
        
        self.set_meta_data()
        print(self._classnames)
    
        # Checks
        if classes_list is not None:
            assert self._num_classes == len(set(classes_list))

    @property
    def targets(self):
        """ For compatibility with AvalancheDataset.
        Adapt targets to appropriate range [0,N]"""
        orig_targets: list = super().targets

        if self.targets_adapter:
            targets = list(map(self.targets_adapter, orig_targets))
        else:
            targets = orig_targets

        return targets

    def __getitem__(self, index):
        """Apply targets adapter on targets."""
        sample, target = super().__getitem__(index)

        if self.targets_adapter:
            target = self.targets_adapter(target)

        return sample, target

    def subset_only_with_labels(self, data_source: List[Datum], labels: list) -> List[Datum]:
        """Return new list only containing Datum from data_source with label in labels."""
        out_list = []
        unique_labels = set(labels)
        '''max_data={}
        for c in unique_labels:
            max_data[c]=0
        for item in data_source:
            if item.label in unique_labels:
                max_data[item.label]+=1
                if max_data[item.label]<81:
                    out_list.append(item)'''
        for item in data_source:
            if item.label in unique_labels:
                out_list.append(item)
        return out_list

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.
        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def datapoints_per_class(self, data_source):
        """Return list of (class,nb_datapoints) tuples."""
        dp_per_class = []
        per_class_dict = self.split_dataset_by_label(data_source)

        for c in sorted(list(per_class_dict.keys())):
            dp_per_class.append((c, len(per_class_dict[c])))

        return dp_per_class
class MiniDomainNet_Aux(DomainNet):
    """
    Subset of less noisy labels from DomainNet.
    Only 4 of the 6 domains are considered. Class-split selected by
    https://arxiv.org/pdf/1904.06487.pdf and https://arxiv.org/pdf/2003.07325.pdf

    For MiniDomainNet splits (4 domains), see: https://drive.google.com/open?id=15rrLDCrzyi6ZY-1vJar3u7plgLe4COL7)
    """
    nb_classes = 10
    domains = ['real']  # No infograph,'clipart', 'painting', 'real', 'quickdraw', 'sketch' painting
    classes_list =  [ 77, 100,  117, 256,270, 282, 291, 293, 294,  344]
    #classes_list=range(344)
    assert len(classes_list) == nb_classes
    """ Subset of less noisy labels, class-split selected by 
    https://arxiv.org/pdf/1904.06487.pdf and https://arxiv.org/pdf/2003.07325.pdf"""

    def __init__(self, classes_list: List[int] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Loads all data

        if classes_list is None:
            classes_list = self.classes_list

        # Keep original labels for summaries, but return adapted (in [0,N] range) label with adapter
        self.orig_to_new_target_map = {old_target: new_target
                                       for new_target, old_target in enumerate(sorted(classes_list))}
        self.targets_adapter = lambda x: int(self.orig_to_new_target_map[int(x)])
        """targets_adapter: Function mapping target to other target. (e.g. transform label-subset to valid
                                [0,N] range). Mapped at __get_item__ and calling the .target property."""

        # Subset the data
        self.data = self.subset_only_with_labels(self.data, classes_list)
        
        self.set_meta_data()
        print(self._classnames)
    

    @property
    def targets(self):
        """ For compatibility with AvalancheDataset.
        Adapt targets to appropriate range [0,N]"""
        orig_targets: list = super().targets

        if self.targets_adapter:
            targets = list(map(self.targets_adapter, orig_targets))
        else:
            targets = orig_targets

        return targets

    def __getitem__(self, index):
        """Apply targets adapter on targets."""
        sample, target = super().__getitem__(index)

        if self.targets_adapter:
            target = self.targets_adapter(target)

        return sample, target

    def subset_only_with_labels(self, data_source: List[Datum], labels: list) -> List[Datum]:
        """Return new list only containing Datum from data_source with label in labels."""
        out_list = []
        unique_labels = set(labels)
        '''max_data={}
        for c in unique_labels:
            max_data[c]=0
        for item in data_source:
            if item.label in unique_labels:
                max_data[item.label]+=1
                if max_data[item.label]<81:
                    out_list.append(item)'''
        for item in data_source:
            if item.label in unique_labels:
                out_list.append(item)
        return out_list

    def split_dataset_by_label(self, data_source):
        """Split a dataset, i.e. a list of Datum objects,
        into class-specific groups stored in a dictionary.
        Args:
            data_source (list): a list of Datum objects.
        """
        output = defaultdict(list)

        for item in data_source:
            output[item.label].append(item)

        return output

    def datapoints_per_class(self, data_source):
        """Return list of (class,nb_datapoints) tuples."""
        dp_per_class = []
        per_class_dict = self.split_dataset_by_label(data_source)

        for c in sorted(list(per_class_dict.keys())):
            dp_per_class.append((c, len(per_class_dict[c])))

        return dp_per_class    
#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from avalanche.benchmarks.datasets import default_dataset_location
from pathlib import Path
from typing import Union, Optional, Any, List
import pandas as pd

from avalanche.benchmarks.classic.classic_benchmarks_utils import \
    check_vision_benchmark
from avalanche.benchmarks import dataset_benchmark
from torchvision import transforms

from collections import defaultdict
from tabulate import tabulate

# See MiniDOmainnet: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/configs/datasets/da/mini_domainnet.yaml
DOMAIN_NET_W = 224
MINI_DOMAIN_NET_W = 96
TINY_DOMAIN_NET_W = MINI_DOMAIN_NET_W
_default_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),  # PIL [0,255] range to [0,1]
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))  # From ImageNet
])


_default_eval_transform = transforms.Compose([
    
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # PIL [0,255] range to [0,1]
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))  # From ImageNet
])

def MiniDomainNetBenchmark(
        *,
        train_transform: Optional[Any] = _default_train_transform,
        eval_transform: Optional[Any] = _default_eval_transform,
        fixed_class_order: Optional[Sequence[int]] = None,
        dataset_root: Union[str, Path],
        seed: Optional[int] = None):
    """
    Creates a CL benchmark using a sequence of 4 MiniDomainNet tasks, where each task is one domain.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default eval transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    if dataset_root is None:
        dataset_root = default_dataset_location('MiniDomainNet')

    considered_classes = MiniDomainNet.classes_list

    # Datasets
    train_sets, test_sets = [], []
    for domain in MiniDomainNet.domains:
        train_sets.append(MiniDomainNet(classes_list=considered_classes, ds_root=dataset_root, domain=domain,
                                        train=True, transform=train_transform))
        test_sets.append(MiniDomainNet(classes_list=considered_classes, ds_root=dataset_root, domain=domain,
                                       train=False, transform=eval_transform))

    
    # TRAINING SUMMARY
    print(f"\n\n {'*' * 40} TRAINING SUMMARY {'*' * 40}")
    produce_class_summary(train_sets)

    # TESTING SUMMARY
    print(f"\n\n {'*' * 40} TESTING SUMMARY {'*' * 40}")
    produce_class_summary(test_sets)
    
    return nc_nd_benchmark(
            train_dataset=train_sets[0],
            test_dataset=test_sets[0],
            train_dataset2=train_sets[1],
            test_dataset2=test_sets[1],
            n_experiences=10,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=False,
            train_transform=None,
            eval_transform=None) 
        




def produce_class_summary(datasets: List[MiniDomainNet]):
    """ Make a per-class summary over all classes in MiniDomainNet."""
    datacnt_per_class = defaultdict(list)  # Datapoints per class (in total)
    datacnt_per_domain = []
    domain_headers = []

    for ds_idx, ds in enumerate(datasets):  # Iterate tasks (domains)
        domain_headers.append(f"#{ds.domain}")
        per_label_dict = ds.split_dataset_by_label(ds.data)

        for c, c_samples in per_label_dict.items():  # Iterate classes with counts
            datacnt_per_class[c].append(len(c_samples))

        # Per-domain summary
        datacnt_per_domain.append(len(ds))

    # Sort classes
    sorted_per_class = [[c_cnt, c, *datacnt_per_class[c]]
                        for c_cnt, c in enumerate(sorted(list(datacnt_per_class.keys())))]

    # Add min-max summary lines
    counts_df = pd.DataFrame.from_records(sorted_per_class)  # Includes class-idxs in first two columns

    # Add summary count as column
    counts_df = counts_df.assign(Total=counts_df.iloc[:, 2:].sum(axis=1))

    col_sum, col_min, col_max = [], [], []
    for column_name in counts_df.iloc[:, 1:]:
        column = counts_df[column_name]
        col_sum.append(column.sum())
        col_min.append(column.min())
        col_max.append(column.max())

    # Append summary line
    totals_row = ['SUM', *col_sum]  # Class'label
    mins_row = ['MIN', *col_min]  # Class'label
    maxs_row = ['MAX', *col_max]  # Class'label

    # Display
    all_rows = [*counts_df.values.tolist(), totals_row, mins_row, maxs_row]
    headers = ['class_idx', 'orig Class', *domain_headers, '#Total']
    print(tabulate(all_rows, headers=headers))


if __name__ == "__main__":
    import sys

    DomainNet_path = ""
    benchmark_instance = MiniDomainNetBenchmark(dataset_root=None)
    check_vision_benchmark(benchmark_instance)
    sys.exit(0)
    
    
__all__ = [
    'DomainNet',
    'MiniDomainNet',
    'MiniDomainNetBenchmark',
    'MiniDomainNet_Aux'
]
