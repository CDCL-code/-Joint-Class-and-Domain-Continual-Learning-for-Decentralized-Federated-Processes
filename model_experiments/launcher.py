from model_experiments.functions_experiment import *
from model_experiments.train_model import main


if __name__ == '__main__':
    args = init_experiment_args()
    gs = MultiRuns(main)
    gs(args)
    gs.wait()
