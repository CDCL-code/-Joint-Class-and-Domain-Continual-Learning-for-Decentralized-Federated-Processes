import copy
import os
import torch
import torch.nn.functional as F
from torch.utils.data import random_split, Dataset
from avalanche.benchmarks import Experience
from avalanche.benchmarks.utils import AvalancheDataset, AvalancheConcatDataset
from avalanche.benchmarks.utils.data_loader import GroupBalancedInfiniteDataLoader
from avalanche.benchmarks.utils.dataset_utils import ConstantSequence
from avalanche.models import avalanche_forward
from avalanche.training.strategies import BaseStrategy
from torch.nn import Module
from torch.optim import Optimizer
import numpy as np

class CDCLDistillation(BaseStrategy):
    def __init__(self, model: Module, optimizer: Optimizer, experiment_scenario, reset_model,
                 loss_type, ce_loss, num_iter_per_exp, **kwargs):
        super().__init__(model, optimizer, **kwargs)
        
        assert loss_type in {'mse', 'kldiv', 'none', 'kldiv_mse', 'LwF_CDCL'}

        # strategy hyperparams
        self.experiment_scenario=experiment_scenario
        self.reset_model = reset_model
        self.loss_type = loss_type
        self.ce_loss = ce_loss
        self.num_iter_per_exp = num_iter_per_exp

        # strategy state
        self.current_exp_num_iters = 0
        self.init_model = None
        self.expert_model = None
        self.prev_model = None
        self.prev_classes = []
        self.logits_target = None
        self.eval_streams = None
        self.list_mse=[]
        self.list_ce=[]

    def _after_forward(self, **kwargs):
        super()._after_forward(**kwargs)
        is_single_task = len(set([el for arr in self.experience.benchmark.task_labels for el in arr])) == 1
        self.logits_targets = torch.zeros_like(self.mb_output)
        curr_classes = self.experience.classes_in_this_experience

        # current model
        self.logits_targets[:, curr_classes] += self.get_masked_bn_logits(self.expert_model, curr_classes)
        

        if self.prev_model is not None:

                self.logits_targets[:, curr_classes] = self.logits_targets[:, curr_classes]
                # model distillation. Combines model targets with previous CL model.
                self.logits_targets[:, self.prev_classes] += self.get_masked_bn_logits(self.prev_model, self.prev_classes)

                intersection = list(set(curr_classes).intersection(self.prev_classes))
                if len(intersection) > 0:
                    
                    self.logits_targets[:, intersection] = 0.5 * self.logits_targets[:, intersection]


    def criterion(self):
        if self.is_training:
            
            ll=0
            
            if  'mse' in self.loss_type:
                ll = self.mse_loss()
                self.list_mse.append(ll.cpu().detach().numpy().item())
            if self.loss_type == 'kldiv':
                ll = self.kldiv_loss()
            if 'LwF_CDCL' in self.loss_type:
                ll= self.lwf_mse_loss()
                self.list_mse.append(ll.cpu().detach().numpy().item())
                cross= F.cross_entropy(self.mb_output, self.mb_y)
                self.list_ce.append(cross.cpu().detach().numpy().item())
                ll += cross
            if self.ce_loss:
                cross= F.cross_entropy(self.mb_output, self.mb_y)
                self.list_ce.append(cross.cpu().detach().numpy().item())
                ll += cross
                
            return ll
        else:
            return F.cross_entropy(self.mb_output, self.mb_y)

    def get_masked_bn_logits(self, model, selected_classes):
        with torch.no_grad():
            curr_logits = avalanche_forward(model, self.mb_x, self.mb_task_id)[:, selected_classes]
        return curr_logits

    def mse_loss(self):
        return F.mse_loss(self.mb_output, self.logits_targets)
    
    def LwF_mse_loss(self):
        results=[]
        if self.prev_model is not None:
            curr_classes = self.experience.classes_in_this_experience
            mb_output2=self.mb_output.clone().detach()
            mb_output2= avalanche_forward(self.prev_model, self.mb_x, self.mb_task_id)
            intersection = list(set(curr_classes).intersection(self.prev_classes))
           
        else:
            results=[]
            for z in self.mb_y:
                res=[]
                for x in range(10):
                    res.append(float(0))
                
                results.append(res)
            results=torch.as_tensor(results, device='cuda')
        mb_output2=results
        return F.mse_loss(self.mb_output, mb_output2)

    def kldiv_loss(self):
        return F.kl_div(self.mb_output, self.logits_targets, reduction='mean', log_target=True)


    def train_dataset_adaptation(self, **kwargs):
        super().train_dataset_adaptation(**kwargs)
        if self.experience_data=='same':
            buffer = self.experience.dataset
        else:
            buffer = self.make_buffer()
        self.adapted_dataset = buffer
        

    def _after_train_dataset_adaptation(self, **kwargs):
        super()._after_train_dataset_adaptation()
        if self.reset_model:
            if self.init_model is None:  # first experience.
                self.init_model = copy.deepcopy(self.model)
            else:
                self.model = copy.deepcopy(self.init_model)

    def _before_training_exp(self, **kwargs):
        if self.clock.train_exp_counter == 0:
            self.model = copy.deepcopy(self.expert_model)
            self.stop_training()

        self.current_exp_num_iters = 0
        super()._before_training_exp(**kwargs)

    def _after_training_exp(self, **kwargs):
        super()._after_training_exp(**kwargs)
        self.prev_model = copy.deepcopy(self.model)
        self.prev_model.eval()
        self.prev_classes.extend(self.experience.classes_in_this_experience)

    def make_buffer(self):
        """ Prepare the data used for the ex-model distillation. """
        assert NotImplementedError()

    def train(self, experiences, eval_streams=None, expert_models=None, **kwargs):

        self.is_training = True
        self.model.train()
        self.model.to(self.device)

        if isinstance(experiences, Experience):
            experiences = [experiences]
        if isinstance(expert_models, torch.nn.Module):
            expert_models = [expert_models]
        if eval_streams is None:
            eval_streams = [experiences]
        for i, exp in enumerate(eval_streams):           
            if isinstance(exp, Experience):
                eval_streams[i] = [exp]
        self.eval_streams = eval_streams

        self._before_training(**kwargs)
        for self.expert_model, self.experience in zip(expert_models, experiences):
            curr_classes = self.experience.classes_in_this_experience
            print('Classes in this experience:')
            print(curr_classes)
            self.expert_model = self.expert_model.to(self.device)
            self.expert_model.eval()
            self.train_exp(self.experience, eval_streams, **kwargs)
            self.expert_model.to('cpu')
        self._after_training(**kwargs)

        res = self.evaluator.get_last_metrics()
        
        
        return res

    def _before_training_iteration(self, **kwargs):
        if self.current_exp_num_iters == self.num_iter_per_exp:
            self.stop_training()
        

        self.current_exp_num_iters += 1
        super()._before_training_iteration(**kwargs)

    def make_train_dataloader(self, num_workers=0, shuffle=True,
                              pin_memory=True, **kwargs):
        """
        Called after the dataset adaptation. Initializes the data loader.
        :param num_workers: number of thread workers for the data loading.
        :param shuffle: True if the data should be shuffled, False otherwise.
        :param pin_memory: If True, the data loader will copy Tensors into CUDA
            pinned memory before returning them. Defaults to True.
        """
        self.dataloader = GroupBalancedInfiniteDataLoader(
            [self.adapted_dataset],
            num_workers=num_workers,
            batch_size=self.train_mb_size,
            pin_memory=pin_memory)


class CDCL(CDCLDistillation):
    def __init__(self, model: Module, optimizer: Optimizer, experience_data, experiment_scenario,
        reset_model, loss_type, ce_loss, **kwargs):

        super().__init__(model, optimizer, experiment_scenario, reset_model, loss_type, ce_loss, **kwargs)
        self.experience_data = experience_data
        print('CDCL\n\n')

    @property
    def mb_y(self):
        if self.is_training:
            return self.logits_targets.argmax(dim=1)
        else:
            return super().mb_y

    def make_buffer(self):
        t = ConstantSequence(0, len(self.experience_data))
        #transform_groups=load_buffer_transform(**kwargs)
        buffer = AvalancheDataset(self.experience_data, task_labels=t)#,transform_groups=transform_groups)
        return buffer
