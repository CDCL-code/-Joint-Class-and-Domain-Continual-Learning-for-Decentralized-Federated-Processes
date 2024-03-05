import os
import torch
import numpy as np
from avalanche.benchmarks.datasets import default_dataset_location

from avalanche.benchmarks.utils import AvalancheDataset, AvalancheConcatDataset
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from torchvision.datasets import FashionMNIST, SVHN, MNIST, USPS
from torchvision import transforms as t
from torchvision.transforms import Normalize, RandomHorizontalFlip, RandomCrop, \
    RandomRotation, CenterCrop, ColorJitter, RandomResizedCrop, Grayscale
from torchvision.transforms import Resize, ToTensor, Compose
from torchvision.models import resnet18, densenet121
from avalanche.benchmarks.classic import PermutedOmniglot, SplitFMNIST
from avalanche.benchmarks import SplitMNIST, SplitMNIST_USPS, SplitMNIST_RotMNIST, SplitMNIST_USPS_extended, SplitMNIST_USPSx2,  MiniDomainNetBenchmark, MiniDomainNet_Aux, DomainNet
from avalanche.benchmarks.classic.ccifar100 import _get_cifar100_dataset, \
    _default_cifar100_eval_transform, _get_cifar10_dataset
from avalanche.benchmarks.classic.cmnist import _get_mnist_dataset
from avalanche.benchmarks.datasets import default_dataset_location, Omniglot
from exmodel.models.models import LeNet5
from torch import nn
from torchvision import transforms
from learn2learn.vision.models import OmniglotCNN,OmniglotFC
import dill as dill
from avalanche.benchmarks.generators import dataset_benchmark
import random
import math
from torch.utils.data import TensorDataset

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

SEED_BENCHMARKS = 1234  # old seed. Don't use it anymore.
SEED_BENCHMARK_RUNS = [1234, 2345, 3456, 5678, 6789]  # 5 different seeds to randomize class orders
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context




def get_aux_data(args):
    if args.aux_data=='same':
        return 'same'
    elif args.aux_data=='mnist':
        dataset_root = default_dataset_location('mnist')
        transforms = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,)), Grayscale(num_output_channels=1)])
        #buffer=SVHN(root=dataset_root, split='test',download=True, transform=transforms)
        buffer=MNIST(root='mnist', train=True, download=True,  transform=transforms)
        t = ConstantSequence(0, len(buffer))
        return AvalancheDataset(buffer, task_labels=t)
    elif 'fashion' in args.aux_data:
        print('fashion')
        trs = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))])
        return FashionMNIST(root=default_dataset_location('fashion_mnist'), download=True, transform=trs)
    elif 'omniglot' in args.aux_data:
        print('omniglot')
        normalize = Normalize(mean=[0.92206], std=[0.08426])
        
        return Omniglot(root=default_dataset_location('omniglot'), transform=Compose([Resize(32),ToTensor(), normalize])) 
    elif args.aux_data in ['painting', 'real', 'clipart']:
        dataset_root = default_dataset_location('MiniDomainNet')
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),  # PIL [0,255] range to [0,1]
            torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))  # From ImageNet
        ])
        considered_classes=None
        return MiniDomainNet_Aux( ds_root=dataset_root, domain=args.aux_data,
                                        train=True, transform=train_transform)
    else:
        raise ValueError(f"No available auxiliary data for scenario {args.scenario}")

def load_buffer_transform(args):
    if 'mnist' in args.scenario or 'multi' in args.scenario or 'split' in args.scenario :
        return Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))])
    elif 'domainnet' in args.scenario:
        return Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomRotation(30),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),  # PIL [0,255] range to [0,1]
            torchvision.transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))  # From ImageNet
        ])
    else:
        raise ValueError(f"No available transform for scenario: {args.scenario}")
                       
                       
def load_scenario(args, run_id):
    CURR_SEED = SEED_BENCHMARK_RUNS[run_id]
    print('Scenario: '+args.scenario)

    if args.model == 'lenet':  # LeNet wants 32x32 images
        transforms = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,))])
    if 'split_mnist' in args.scenario:
        scenario = SplitMNIST(n_experiences=5, return_task_id=False, train_transform=transforms, eval_transform=transforms, seed=CURR_SEED)
    elif args.scenario=='domainnet':
        scenario=MiniDomainNetBenchmark(dataset_root=None)
    elif 'multi_digitx2' in args.scenario:
        transforms = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,)), Grayscale(num_output_channels=1)])
        
        if args.random:
            scenario=SplitMNIST_USPSx2(n_experiences=10,return_task_id=False, train_transform=transforms,
                                       eval_transform=transforms, seed=CURR_SEED)
        else:
            scenario=SplitMNIST_USPSx2(n_experiences=10,return_task_id=False, train_transform=transforms, eval_transform=transforms)
    elif 'multi_digit' in args.scenario:
        transforms = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,)), Grayscale(num_output_channels=1)])
        if args.random:
            scenario=SplitMNIST_USPS(n_experiences=10,return_task_id=False, train_transform=transforms, eval_transform=transforms,seed=run_id)
        else:
            scenario=SplitMNIST_USPS(n_experiences=10,return_task_id=False, train_transform=transforms, eval_transform=transforms)
    
    elif args.scenario == 'multi_mnist':
        transforms = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,)), Grayscale(num_output_channels=1)])
        scenario=SplitMNIST_RotMNIST(n_experiences=10,return_task_id=False, train_transform=transforms, eval_transform=transforms,seed=run_id)
    
    elif args.scenario == 'joint_mnist': #to train local models
        scenario = SplitMNIST(n_experiences=1, return_task_id=False, seed=CURR_SEED, train_transform=transforms, eval_transform=transforms)
        
    elif args.scenario == 'Fashion_mnist':
        scenario =  SplitFMNIST(n_experiences=10, train_transform=transforms, eval_transform=transforms)
    elif 'split_rot_mnist' in args.scenario : #to train local models
        transforms = Compose([Resize(32), ToTensor(), Normalize((0.1307,), (0.3081,)), Grayscale(num_output_channels=1)])
        scenario =  Rotated_SplitMNIST(n_experiences=5, train_transform=transforms, eval_transform=transforms, seed=run_id)
    
    elif 'domainnet' in args.scenario:
        scenario=MiniDomainNetBenchmark(dataset_root=None, seed=run_id)      
    else:
        raise ValueError(f"Unknown scenario name: {args.scenario}")
    
    return scenario


def load_model(args) -> nn.Module:

    num_classes = args.num_classes
    
    orig_file = os.path.join(args.logdir, args.model, args.scenario, 'orig_model.pt')
    if os.path.exists(orig_file):
        print("loading from fixed initialization.")
        return torch.load(orig_file)
    
    if args.model == 'lenet':
        input_channels = 1 if 'mnist' or 'omniglot' or 'multi' in args.scenario else 3
        return LeNet5(num_classes , input_channels)
    elif args.model == 'dense':
        model = densenet121(pretrained=False)
        #the dense network is in a library and take imput with 3 channel, the next command change it to take 1 channel input
        #model.features[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    elif args.model == 'resnet':
        num_classes=args.num_classes
        return resnet18(num_classes=numclasses)
    elif args.model == 'resnet_pretrained':
        return resnet18(pretrained=True)
    
    elif args.model == 'omni':
        model=OmniglotCNN(output_size=10)
        return model
    else:
        raise ValueError("Wrong model name.")


def load_exmodels(args, scenario, run_id):
    curr_scenario = args.scenario
    expert_arch = args.model

    if args.pret == 'Fashion_mnist' or 'split_mnist':
        curr_scenario = args.pret
    if args.pret == 'joint_mnist':
        curr_scenario = args.pret
    if 'svhn' in curr_scenario:
        expert_arch = 'dense'
    try:
        models = []
        print('Models from experiences: ')
        for curr_sc in args.datasets.split(', '):
                if 'multi_digit' in curr_sc:
                    for i in range(args.n_nodes):
                        try:                  
                            run=run_id                            
                            model_fname = os.path.join('cdcl/ex_model_cl/logs/pret_models', expert_arch, curr_sc, f'run{run}', f'model_e{i}.pt')
                            print(model_fname)
                            model = torch.load(model_fname).to('cpu')
                            model.eval()
                            models.append(model)
                        except:
                            #try with run 0, this means that local models have only one run
                            model_fname = os.path.join('cdcl/ex_model_cl/logs/pret_models', expert_arch, curr_sc, f'run0', f'model_e{i}.pt')
                            print(model_fname)
                            model = torch.load(model_fname).to('cpu')
                            model.eval()
                            models.append(model)
                else:
                    for i in range(int(args.n_nodes/args.n_domains)):                   
                        try:                  
                            model_fname = os.path.join('cdcl/ex_model_cl/logs/pret_models', expert_arch, curr_sc, f'run{run_id}', f'model_e{i}.pt')
                            print(model_fname)
                            model = torch.load(model_fname).to('cpu')
                            model.eval()
                            models.append(model)
                        except:
                            model_fname = os.path.join('cdcl/ex_model_cl/logs/pret_models', expert_arch, curr_sc, f'run0', f'model_e{i}.pt')
                            print(model_fname)
                            model = torch.load(model_fname).to('cpu')
                            model.eval()
                            models.append(model)
    
    except FileNotFoundError:
        print(f"File not found: {model_fname}")
        print("please train local models before.")
        raise
    if args.random:
        order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        random.seed(run_id)
        random.shuffle(order)
        print('Order experiences:')
        print(order)
        models=[models[i] for i in order]
    return models
# Changed basestring to str, and dict uses items() instead of iteritems().
def to_json(o, level=0, ignore_errors=False):
    """ pretty-print json.
    source: https://stackoverflow.com/questions/10097477/python-json-array-newlines
    :param o:
    :param level:
    :return:
    """
    INDENT = 3
    SPACE = " "
    NEWLINE = "\n"
    ret = ""
    if isinstance(o, dict):
        ret += "{" + NEWLINE
        comma = ""
        for k, v in o.items():
            ret += comma
            comma = ",\n"
            ret += SPACE * INDENT * (level + 1)
            ret += '"' + str(k) + '":' + SPACE
            ret += to_json(v, level + 1, ignore_errors)

        ret += NEWLINE + SPACE * INDENT * level + "}"
    elif isinstance(o, str):
        ret += '"' + o + '"'
    elif isinstance(o, list):
        ret += "[" + ",".join([to_json(e, level + 1, ignore_errors) for e in o]) + "]"
    # Tuples are interpreted as lists
    elif isinstance(o, tuple):
        ret += "[" + ",".join(to_json(e, level + 1, ignore_errors) for e in o) + "]"
    elif isinstance(o, bool):
        ret += "true" if o else "false"
    elif isinstance(o, int):
        ret += str(o)
    elif isinstance(o, float):
        ret += '%.7g' % o
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.integer):
        ret += "[" + ','.join(map(str, o.flatten().tolist())) + "]"
    elif isinstance(o, numpy.ndarray) and numpy.issubdtype(o.dtype, numpy.inexact):
        ret += "[" + ','.join(map(lambda x: '%.7g' % x, o.flatten().tolist())) + "]"
    elif o is None:
        ret += 'null'
    elif ignore_errors:
        # we do not recognize the type but we don't want to raise an error.
        ret = '"<not serializable>"'
    else:
        # Unknown type. Raise error.
        raise TypeError("Unknown type '%s' for json serialization" % str(type(o)))
    return ret
        
def train_loop(log_dir, scenario, strategy, peval_on_test=True):
    
    for idx, experience in enumerate(scenario.train_stream):
        print('Experience n: '+str(idx))
        if peval_on_test:         
            evals = [scenario.test_stream]
        else:
            evals = [[experience]]
        strategy.train(
            experiences=experience,
            eval_streams=evals,
            expert_models=scenario.trained_models[idx],
            pin_memory=True, num_workers=8)
        model_fname = os.path.join(log_dir,
                                   f'model_e{experience.current_experience}.pt')
        torch.save(strategy.model, model_fname)

        strategy.eval(scenario.train_stream[:])
        strategy.eval(scenario.test_stream[:])
        
        
    #save the results
    with open(os.path.join(log_dir, 'metrics.json'), 'w') as f:
        f.write(to_json(strategy.evaluator.get_all_metrics(), ignore_errors=True))
    with open(os.path.join(log_dir, 'mse.json'), 'w') as f:
            f.write(to_json(strategy.list_mse, ignore_errors=True))
    with open(os.path.join(log_dir, 'ce.json'), 'w') as f:
            f.write(to_json(strategy.list_ce, ignore_errors=True))
    with open(os.path.join(log_dir, 'early_stopping_epochs.json'), 'w') as f:
        f.write(to_json(strategy.early_stopping_epoch, ignore_errors=True))
    
            
