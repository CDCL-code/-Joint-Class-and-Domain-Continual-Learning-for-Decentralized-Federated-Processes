"""
    Inspired by Carta, A., Cossu, A., Lomonaco, V., & Bacciu, D. (2021). 
    Ex-Model: Continual Learning from a Stream of Trained Models. arXiv preprint arXiv:2112.06511.
"""
import os

from torch.utils.data import random_split

from model_experiments.strategy_exml_distillation import CDCL
import torch
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, \
    timing_metrics, forgetting_metrics

from avalanche.training.plugins import EvaluationPlugin, EarlyStoppingPlugin

from model_experiments.utils import *
from model_experiments.functions_experiment import *

def train(args):
    log_dir = args.logdir
    train_epochs=args.train_epochs

    # check if selected GPU is available or use CPU
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    #device='cpu'
    print(f'Using device: {device}')
    # create scenario
    scenario = load_scenario(args, args.run_id)
    models = load_exmodels(args, scenario, args.run_id)
    scenario = ModelScenario(scenario, models)
    buffer_transform = load_buffer_transform(args)

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True, minibatch=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        forgetting_metrics(experience=True, stream=True))

    model = load_model(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    pgn=[]
    if args.early_stopping:
        pgn.append(EarlyStoppingPlugin(patience=args.early_stopping, val_stream_name='test_stream/Task000'))
    kwargs = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'device': device,
        'train_mb_size': args.train_mb_size,
        'eval_mb_size': 24,
        'evaluator': eval_plugin,
        'eval_every': 1,
        'reset_model': args.reset_model,
        'loss_type': args.loss_type,
        'ce_loss': args.ce_loss,
        'plugins': pgn,  # [ProfilerPlugin()],
        'num_iter_per_exp': args.num_iter_per_exp,
        'experiment_scenario': args.experiment_scenario
    }
    experience_data = get_aux_data(args)
    strat_args = {'experience_data': experience_data}
    strategy = CDCL(**strat_args, **kwargs, train_epochs=args.train_epochs)
   
    # train on the selected scenario with the chosen strategy
    print('Starting experiment...')
    peval_on_test = args.peval_on_test
    train_loop(log_dir, scenario, strategy, peval_on_test=peval_on_test)


def main(args):
    train(args)
