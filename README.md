# Joint Class and Domain Continual Learning for Decentralized Federated Processes

We propose an approach using  knowledge distillation (KD) to share knowledge extracted from local models trained with the data at each node and to incrementally generate a global model



## Install Dependencies
```
conda env create -f environment.yml
```
avalanche is added here since new classes and function. In particular

##### in benchmarks/scenarios/new_classes/nc_scenario.py
NCNDScenario : two domains with same classes, different domains will have same classes in the elabotaion
NCNDScenariox2: two domains with same classes, different domains will have same classes in the elabotaion, but the combinations of classes-domain are repeated (i.e. scenario 3)
NCNDScenario3: three domains with same classes, different domains will have same classes in the elabotaion


##### in benchmarks/generators/benchmark_generators.py
nc_nd_benchmark
nc_nd_benchmarkx2
nc_nd_benchmark3

##### in benchmarks/classic/cmnist.py
SplitMNIST_USPS
SplitMNIST_USPSx2
SplitMNIST_RotMNIST

##### in benchmarks/classic/domainnet.py
MiniDomainNet
MiniDomainNetBenchmark
MiniDomainNet_Aux

## Run Experiments
to launch an experiment run:
```
python experiments/launcher.py --config  CONFIGS/12/[name of the file].yaml
```
The directory `CONFIGS` contains the configuration of some of the experiment ran


to train the local models:
```
python model_experiments/prepare_pretrained_models.py --model lenet --scenario split_mnist --epochs 5 --lr 0.01
```

to visualize the results of the experiment, use the notebook visualize.ipynp