################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 26-06-2020                                                             #
# Author(s): Gabriele Graffieti                                                #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################
from pathlib import Path
from albumentations.pytorch.transforms import ToTensorV2
from albumentations import *
from typing import Optional, Sequence, Union, Any
import torch
from PIL.Image import Image
import copy
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import MNIST, SVHN, USPS
from torchvision.transforms import ToTensor, ToPILImage, Compose, Normalize, \
    RandomRotation,RandomCrop
from torch.utils.data import TensorDataset
import numpy as np
from avalanche.benchmarks.generators import dataset_benchmark
from avalanche.benchmarks import NCScenario, nc_benchmark, ni_benchmark, nc_nd_benchmark, NCNDScenario, nc_nd_benchmarkx2,nc_nd_benchmark3
from avalanche.benchmarks.classic.classic_benchmarks_utils import \
    check_vision_benchmark
from avalanche.benchmarks.datasets import default_dataset_location
from avalanche.benchmarks.utils import AvalancheDataset

_default_mnist_train_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])

_default_mnist_eval_transform = Compose([
    ToTensor(),
    Normalize((0.1307,), (0.3081,))
])


class PixelsPermutation(object):
    """
    Apply a fixed permutation to the pixels of the given image.

    Works with both Tensors and PIL images. Returns an object of the same type
    of the input element.
    """

    def __init__(self, index_permutation: Sequence[int]):
        self.permutation = index_permutation
        self._to_tensor = ToTensor()
        self._to_image = ToPILImage()

    def __call__(self, img: Union[Image, Tensor]):
        is_image = isinstance(img, Image)
        if (not is_image) and (not isinstance(img, Tensor)):
            raise ValueError('Invalid input: must be a PIL image or a Tensor')

        if is_image:
            img = self._to_tensor(img)

        img = img.view(-1)[self.permutation].view(*img.shape)

        if is_image:
            img = self._to_image(img)

        return img

def SplitMNIST_USPS_extended(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):
    task_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]

    USPS_train, USPS_test = _get_USPS_dataset(dataset_root)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)
    
    print(len(mnist_test))
    mnist_test, mnist_test2 = torch.utils.data.random_split(mnist_test, [len(USPS_test), len(mnist_test)-len(USPS_test)])
    print(len(mnist_test))
    #train_dataset=split_dataset_by_labels([mnist_train], task_labels)
    #test_dataset=split_dataset_by_labels([mnist_test], task_labels)
    

    if return_task_id:
        return nc_benchmark(
            train_dataset=[mnist_train,USPS_train],
            test_dataset=[mnist_test,USPS_test],
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_nd_benchmark_extended(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            train_dataset2=USPS_train,
            test_dataset2=USPS_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=False,
            train_transform=train_transform,
            eval_transform=eval_transform)
    
    
def SplitMNIST_USPS(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):
    task_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]

    USPS_train, USPS_test = _get_USPS_dataset(dataset_root)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)
    
    print(len(mnist_test))
    mnist_test, mnist_test2 = torch.utils.data.random_split(mnist_test, [len(USPS_test), len(mnist_test)-len(USPS_test)])
    print(len(mnist_test))
    #train_dataset=split_dataset_by_labels([mnist_train], task_labels)
    #test_dataset=split_dataset_by_labels([mnist_test], task_labels)
    print(seed)
    if return_task_id:
        return nc_benchmark(
            train_dataset=[mnist_train,USPS_train],
            test_dataset=[mnist_test,USPS_test],
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_nd_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            train_dataset2=USPS_train,
            test_dataset2=USPS_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=False,
            train_transform=train_transform,
            eval_transform=eval_transform)
    
def SplitMNIST_USPSx2(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):
    task_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]

    USPS_train, USPS_test = _get_USPS_dataset(dataset_root)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)
    
    print(len(mnist_test))
    mnist_test, mnist_test2 = torch.utils.data.random_split(mnist_test, [len(USPS_test), len(mnist_test)-len(USPS_test)])
    print(len(mnist_test))
    #train_dataset=split_dataset_by_labels([mnist_train], task_labels)
    #test_dataset=split_dataset_by_labels([mnist_test], task_labels)
    
    if return_task_id:
        return nc_benchmark(
            train_dataset=[mnist_train,USPS_train],
            test_dataset=[mnist_test,USPS_test],
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_nd_benchmarkx2(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            train_dataset2=USPS_train,
            test_dataset2=USPS_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=False,
            train_transform=train_transform,
            eval_transform=eval_transform)
    
    

    
    
def SplitMNIST_RotMNIST(
            n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):
    
    print(seed)
    rotation_angles=[20,40,60,80]
    rotation_angle=rotation_angles[seed]
    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)
    
    
    rotation = RandomRotation(degrees=(rotation_angle, rotation_angle))
    print(rotation)
    rotation_transforms = dict(
            train=(rotation, None),
            eval=(rotation, None)
    )

    # Freeze the rotation
    rotated_train = AvalancheDataset(
        mnist_train,
        transform_groups=rotation_transforms,
        initial_transform_group='train').freeze_transforms()

    rotated_test = AvalancheDataset(
        mnist_test,
        transform_groups=rotation_transforms,
        initial_transform_group='eval').freeze_transforms()
    

    if return_task_id:
        return nc_benchmark(
            train_dataset=[mnist_train,rotated_train],
            test_dataset=[mnist_test,rotated_test],
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_nd_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            train_dataset2=rotated_train,
            test_dataset2=rotated_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=False,
            fixed_class_order=fixed_class_order,
            shuffle=False,
            train_transform=train_transform,
            eval_transform=eval_transform) 
    
  
    
    
def SplitMNIST_PERMUTED_MNIST(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = True,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):
    task_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]
    """
    Creates a CL benchmark using the MNIST dataset.

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

    :param n_experiences: The number of incremental experiences in the current
        benchmark.
        The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to false.
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
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    USPS_train, USPS_test = _get_USPS_dataset(dataset_root)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)
    
    list_train_dataset = []
    list_test_dataset = []
    rng_permute = np.random.RandomState(seed)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)

    # for every incremental experience
    for _ in range(n_experiences):
        # choose a random permutation of the pixels in the image
        idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(
            torch.int64)

        permutation = PixelsPermutation(idx_permute)

        permutation_transforms = dict(
            train=(permutation, None),
            eval=(permutation, None)
        )

        # Freeze the permutation
        permuted_train = AvalancheDataset(
            mnist_train,
            transform_groups=permutation_transforms,
            initial_transform_group='train').freeze_transforms()

        permuted_test = AvalancheDataset(
            mnist_test,
            transform_groups=permutation_transforms,
            initial_transform_group='eval').freeze_transforms()

        list_train_dataset.append(permuted_train)
        list_test_dataset.append(permuted_test)
        
    mnist_test, mnist_test2 = torch.utils.data.random_split(mnist_test, [len(USPS_test), len(mnist_test)-len(USPS_test)])
    
    

    if return_task_id:
        return nc_benchmark(
            train_dataset=[mnist_train,USPS_train],
            test_dataset=[mnist_test,USPS_test],
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_nd_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            train_dataset2=list_test_dataset,
            test_dataset2=list_test_dataset,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=False,
            train_transform=train_transform,
            eval_transform=eval_transform)

def SplitMNIST(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = False,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):
    """
    Creates a CL benchmark using the MNIST dataset.

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

    :param n_experiences: The number of incremental experiences in the current
        benchmark.
        The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to false.
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
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)
    if return_task_id:
        return nc_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=mnist_train,
            test_dataset=mnist_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)
    
    
def SplitSVHN(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = False,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):


    svhn_train, svhn_test = _get_SVHN_dataset(dataset_root)
    if return_task_id:
        return nc_benchmark(
            train_dataset=svhn_train,
            test_dataset=svhn_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        return nc_benchmark(
            train_dataset=svhn_train,
            test_dataset=svhn_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)

def SplitUSPS(
        n_experiences: int,
        *,
        return_task_id=False,
        seed: Optional[int] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        shuffle: bool = False,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None):
    """
    Creates a CL benchmark using the MNIST dataset.

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

    :param n_experiences: The number of incremental experiences in the current
        benchmark.
        The value of this parameter should be a divisor of 10.
    :param return_task_id: if True, a progressive task id is returned for every
        experience. If False, all experiences will have a task ID of 0.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param fixed_class_order: A list of class IDs used to define the class
        order. If None, value of ``seed`` will be used to define the class
        order. If non-None, ``seed`` parameter will be ignored.
        Defaults to None.
    :param shuffle: If true, the class order in the incremental experiences is
        randomly shuffled. Default to false.
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
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    USPS_train, USPS_test = _get_USPS_dataset(dataset_root)
    print(len(USPS_test.dataset))
    if return_task_id:
        return nc_benchmark(
            train_dataset=USPS_train,
            test_dataset=USPS_test,
            n_experiences=n_experiences,
            task_labels=True,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            class_ids_from_zero_in_each_exp=True,
            train_transform=train_transform,
            eval_transform=eval_transform)
    else:
        
        return nc_benchmark(
            train_dataset=USPS_train,
            test_dataset=USPS_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform)

def PermutedMNIST(
        n_experiences: int,
        *,
        seed: Optional[int] = None,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None) -> NCScenario:
    """
    Creates a Permuted MNIST benchmark.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    Random pixel permutations are used to permute the MNIST images in
    ``n_experiences`` different manners. This means that each experience is
    composed of all the original 10 MNIST classes, but the pixel in the images
    are permuted in a different way.

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    A progressive task label, starting from "0", is applied to each experience.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param n_experiences: The number of experiences (tasks) in the current
        benchmark. It indicates how many different permutations of the MNIST
        dataset have to be created.
        The value of this parameter should be a divisor of 10.
    :param seed: A valid int used to initialize the random number generator.
        Can be None.
    :param train_transform: The transformation to apply to the training data
        before the random permutation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data
        before the random permutation, e.g. a random crop, a normalization or a
        concatenation of different transformations (see torchvision.transform
        documentation for a comprehensive list of possible transformations).
        If no transformation is passed, the default test transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location for 'mnist' will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """

    list_train_dataset = []
    list_test_dataset = []
    rng_permute = np.random.RandomState(seed)

    mnist_train, mnist_test = _get_mnist_dataset(dataset_root)

    # for every incremental experience
    for _ in range(n_experiences):
        # choose a random permutation of the pixels in the image
        idx_permute = torch.from_numpy(rng_permute.permutation(784)).type(
            torch.int64)

        permutation = PixelsPermutation(idx_permute)

        permutation_transforms = dict(
            train=(permutation, None),
            eval=(permutation, None)
        )

        # Freeze the permutation
        permuted_train = AvalancheDataset(
            mnist_train,
            transform_groups=permutation_transforms,
            initial_transform_group='train').freeze_transforms()

        permuted_test = AvalancheDataset(
            mnist_test,
            transform_groups=permutation_transforms,
            initial_transform_group='eval').freeze_transforms()

        list_train_dataset.append(permuted_train)
        list_test_dataset.append(permuted_test)

    return nc_benchmark(
        list_train_dataset,
        list_test_dataset,
        n_experiences=len(list_train_dataset),
        task_labels=True,
        shuffle=False,
        class_ids_from_zero_in_each_exp=True,
        one_dataset_per_exp=True,
        train_transform=train_transform,
        eval_transform=eval_transform)


def Rotated_SplitMNIST(
        n_experiences: int,
        *,
        seed: Optional[int] = None,
        rotations_list: Optional[Sequence[int]] = None,
        fixed_class_order: Optional[Sequence[int]] = None,
        train_transform: Optional[Any] = _default_mnist_train_transform,
        eval_transform: Optional[Any] = _default_mnist_eval_transform,
        dataset_root: Union[str, Path] = None) -> NCScenario:

        print(seed)
        mnist_train, mnist_test = _get_mnist_dataset(dataset_root)
        rotation_angles=[20,40,60,80]
        rotation_angle=rotation_angles[seed]

        rotation = RandomRotation(degrees=(rotation_angle, rotation_angle))
        print(rotation)
        rotation_transforms = dict(
            train=(rotation, None),
            eval=(rotation, None)
        )

        # Freeze the rotation
        rotated_train = AvalancheDataset(
            mnist_train,
            transform_groups=rotation_transforms,
            initial_transform_group='train').freeze_transforms()

        rotated_test = AvalancheDataset(
            mnist_test,
            transform_groups=rotation_transforms,
            initial_transform_group='eval').freeze_transforms()


        return nc_benchmark(
            train_dataset=rotated_train,
            test_dataset=rotated_test,
            n_experiences=n_experiences,
            task_labels=False,
            seed=seed,
            fixed_class_order=fixed_class_order,
            shuffle=False,
            train_transform=train_transform,
            eval_transform=eval_transform)


def _get_mnist_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location('mnist')

    train_set = MNIST(root=dataset_root, train=True, download=True)

    test_set = MNIST(root=dataset_root,
                     train=False, download=True)

    return train_set, test_set


def _get_SVHN_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location('SVHN')

    train_set = SVHN(root=dataset_root,
                      split='train',download=True)

    test_set = SVHN(root=dataset_root,
                     split='test',  download=True)

    return train_set, test_set

def _get_USPS_dataset(dataset_root):
    if dataset_root is None:
        dataset_root = default_dataset_location('USPS')

    train_set = USPS(root=dataset_root,
                      train=True,download=True)

    test_set = USPS(root=dataset_root,
                     train=False,  download=True)

    return train_set, test_set


__all__ = [
    'SplitMNIST',
    'SplitUSPS',
    'PermutedMNIST',
    'Rotated_SplitMNIST',
    'SplitMNIST_USPS',
    'SplitMNIST_PERMUTED_MNIST',
    'SplitSVHN',
    'SplitMNIST_RotMNIST',
    'SplitMNIST_USPS_extended',
    'SplitMNIST_USPSx2'
]


if __name__ == "__main__":
    import sys

    print('Split MNIST')
    benchmark_instance = SplitMNIST(
        5, train_transform=None, eval_transform=None)
    check_vision_benchmark(benchmark_instance)

    print('Permuted MNIST')
    benchmark_instance = PermutedMNIST(
        5, train_transform=None, eval_transform=None)
    check_vision_benchmark(benchmark_instance)

    print('Rotated MNIST')
    benchmark_instance = RotatedMNIST(
        5, train_transform=None, eval_transform=None)
    check_vision_benchmark(benchmark_instance)

    sys.exit(0)
