import argparse
import copy
import torch
import sys
import os
import ray
from model_experiments.utils import *

import re
import yaml


def write_config_file(args, result_folder):
    #function from Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    """
    Write yaml configuration file inside result folder
    """
    os.makedirs(result_folder, exist_ok=True)
    result_folder=args.logdir
    with open(os.path.join(result_folder, 'config_file.yaml'), 'w') as f:
        yaml.dump(dict(vars(args)), f)

class YAMLConfig:
    #Class from Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    def __init__(self, config_file):
        self.config_files = [config_file]
        self._parse_config(config_file)

    def config_name(self):
        s = []
        for el in self.config_files:
            # drop the folder, keep only the filename
            el = el.split('/')[-1].split('.')[0]
            s.append(el)
        return '_'.join(s)

    def update(self, new_config_file):
        """ Parse yaml file """
        self.config_files.append(new_config_file)
        self._parse_config(new_config_file)

    def _parse_config(self, config_file):
        """
        Parse yaml file containing also math notation like 1e-4
        """
        # fix to enable scientific notation
        # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
        loader = yaml.SafeLoader
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))

        with open(config_file, 'r') as f:
            configs = yaml.load(f, Loader=loader)

        for k, v in configs.items():
            self.__dict__[k] = v

    def __str__(self):
        res = 'CONFIG:\n'
        for k, v in self.__dict__.items():
            res += f'\t({k}) -> {v}\n'
        return res

    def __eq__(self, other):
        da = self.__dict__
        db = other.__dict__

        ignore_keys = {'config_file'}
        def check_inclusion_recursive(config_a, config_b):
            for k, v in da.items():
                if k == 'config_files':
                    continue  # we don't care about filenames, they can be different.
                if k in db and db[k] == v:
                    continue
                else:
                    return False
            return True
        return check_inclusion_recursive(da, db) and check_inclusion_recursive(db, da)

class Tee(object):
    #Class from Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    def __init__(self, *files):
        self.files = files
        self.fileno = files[0].fileno

    def write(self, obj):
        for f in self.files:
            if not f.closed:
                f.write(obj)

    def flush(self):
        for f in self.files:
            if not f.closed:
                f.flush()

    @property
    def closed(self):
        return False
    
class Experiment:
    #inspired by Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    def __init__(self, results_fname=None, resume_ok=False, num_cpus=0, num_gpus=0):

        self.results_fname = results_fname
        self.resume_ok = resume_ok
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus

        self.f = None
        self._obj_id = None
        self._done = None

    def __call__(self, config):
        assert not self._done  # can call only once.
        logdir = 'chiara/'

        config_fname = os.path.join(logdir, 'config_file.yaml')

        def run_exp(argum):
            self.f = open(os.path.join(logdir, 'out.txt'), 'w', buffering=1)
            sys.stdout = Tee(sys.stdout, self.f)
            sys.stderr = sys.stdout
            self.run(argum)

        self.f = open(os.path.join(logdir, 'out.txt'), 'w', buffering=1)
        sys.stdout = Tee(sys.stdout, self.f)
        sys.stderr = sys.stdout
        self._obj_id = self.run(config)

    def wait(self):
        if self._obj_id is not None:
            ray.get(self._obj_id)
            if self.f is not None:
                self.f.close()
        self._done = True

    def run(self, config):
        assert NotImplementedError()

    def result(self):
        assert NotImplementedError()
     
        
        
class SingleRunExp(Experiment):
    #Class inspired by Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    def __init__(self, main, num_cpus=0, num_gpus=0):
        super().__init__('metrics.json', resume_ok=True,
                         num_cpus=num_cpus, num_gpus=num_gpus)
        self.main = main

    def run(self, config):
        if config.cuda:
            print(f'Using GPUs {os.environ["CUDA_VISIBLE_DEVICES"]}')
        else:
            print('Using CPUs')
        return self.main(config)




class MultiRuns(Experiment):
    #Class inspired by Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    def __init__(self, main):
        super().__init__(None, resume_ok=True)
        self.main = main

    def run(self, config):
        config.run_id = 0
        orig_args = copy.deepcopy(config)

        # Assessment
        assess_args = []
        print(orig_args.assess_runs)
        for i in range(orig_args.assess_runs):
            print('run here')
            single_config = copy.deepcopy(orig_args)
            single_config.run_id = i
            single_config.logdir = os.path.join(orig_args.logdir, f'ASSESS{i}')
            assess_args.append(single_config)
        assess_exp = SingleRunExp(self.main, num_cpus=orig_args.cpus_per_job,
                                  num_gpus=orig_args.gpus_per_job)
        run_configs_and_wait(assess_exp, assess_args)

        
        
        
def run_configs_and_wait(base_exp, configs, stagger=None):
    #Class inspired by Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    rem_ids = []
    for config in configs:
        exp = copy.deepcopy(base_exp)
        exp(config)
        ids = exp._obj_id

        if ids is not None:
            rem_ids.append(ids)
            if stagger is not None:
                sleep(stagger)
    n_jobs = len(rem_ids)

    print(f"Scheduled jobs: {n_jobs}")

    while rem_ids:
        done_ids, rem_ids = ray.wait(rem_ids, num_returns=1)
        for result_id in done_ids:
            ray.get(result_id)
            n_jobs -= 1
            print(f'Job {result_id} terminated. Jobs left: {n_jobs}')
            
            
            
def init_experiment_args():
    
    #Function inspired by Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    global args, args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='',
                        help='path to yaml configuration file')
    cmd_args = parser.parse_args()
    if cmd_args.config_file == '':
        raise ValueError('You must provide a config file.')
    args = YAMLConfig('./CONFIGS/default.yaml')
    args.update(cmd_args.config_file)
    # expand logdir name
    if args.debug:
        args.logdir = 'chiara/'
    args.logdir = 'chiara/'
    args.logdir = os.path.expanduser(args.logdir)
    config_name = args.config_files[-1].split('/')[-1].split('.')[0]
    args.logdir = os.path.join(args.logdir, config_name)
    os.makedirs(args.logdir, exist_ok=True)
    torch.set_num_threads(args.max_cpus)
    if args.cuda:
        set_gpus(args.max_gpus)
    return args

def set_gpus(num_gpus):
    
    #Function inspired by Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    #Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
    try:
        import gpustat
    except ImportError:
        print("gpustat module is not installed. No GPU allocated.")

    try:
        selected = []

        stats = gpustat.GPUStatCollection.new_query()

        for i in range(num_gpus):

            ids_mem = [res for res in map(lambda gpu: (int(gpu.entry['index']),
                                          float(gpu.entry['memory.used']) /\
                                          float(gpu.entry['memory.total'])),
                                      stats) if str(res[0]) not in selected]

            if len(ids_mem) == 0:
                # No more gpus available
                break

            best = min(ids_mem, key=lambda x: x[1])
            bestGPU, bestMem = best[0], best[1]
            # print(f"{i}-th best GPU is {bestGPU} with mem {bestMem}")
            selected.append(str(bestGPU))

        print("Setting GPUs to: {}".format(",".join(selected)))
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ",".join(selected)
    except BaseException as e:
        print("GPU not available: " + str(e))
        
        
class ModelScenario:
    def __init__(self, original_scenario, trained_models) -> None:
        """
        Class used in Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
        Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511. Ex-model scenario.
        Each experience is a quadruple
        <original_exp, trained_model, generator, buffer>.

        The original experience should NOT BE USED DURING TRAINING.
        Instead, data must be extracted from the original model, generator,
        or buffers.

        :param original_scenario:
        :param trained_models: list of filenames
        """
        mm = []
        for model in trained_models:
            model = model.eval().to('cpu')
            mm.append(model)

        self.scenario = original_scenario
        self.trained_models = mm
        self.train_stream = original_scenario.train_stream
        self.test_stream = original_scenario.test_stream

        




        
        
__all__ = [
    'write_config_file',
    'Experiment',
    'SingleRunExp',
    'Tee',
    'run_configs_and_wait',
    'YAMLConfig',
    'MultiRuns',
    'init_experiment_args',
    'ModelScenario'
]
        