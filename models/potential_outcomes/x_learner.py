import torch
from torch import nn
from torch.distributed.tensor.parallel import loss_parallel

from models.wrapper import Wrapper
from models.potential_outcomes.s_learner import S_learner
from models.potential_outcomes.t_learner import T_learner
from models.potential_outcomes.regressors_models import Treatment_Model

'''
X-learner from "Meta-learners for Estimating Heterogeneous Treatment Effects using Machine Learning" by Künzel et al. 2019
'''


class X_learner(Wrapper):
    def __init__(self, network_name, input_dim,
                 hidden_sizes_outcome, outcome_dist, activation_outcome, dropout_outcome,
                 hidden_sizes_ite, ite_dist, activation_ite, dropout_ite,
                 hidden_sizes_propensity, activation_propensity, dropout_propensity,
                 num_treatments,
                 experiment, model_path,
                 device,
                 weight_init=None,
                 save=True, limit_variance=False, fix_variance=True, limit_shape=False, optim_config=None, wandblog=False):

        super(X_learner, self).__init__('x-learner', device, experiment, model_path, save, optim_config, wandblog)

        self.network_name = network_name
        self.input_dim = input_dim
        self.hidden_sizes_outcome = hidden_sizes_outcome
        self.outcome_dist = outcome_dist
        self.activation_outcome = activation_outcome
        self.dropout_outcome = dropout_outcome

        self.hidden_sizes_ite = hidden_sizes_ite
        self.ite_dist = ite_dist
        self.activation_ite = activation_ite
        self.dropout_ite = dropout_ite

        self.hidden_sizes_propensity = hidden_sizes_propensity
        self.activation_propensity = activation_propensity
        self.dropout_propensity = dropout_propensity

        self.num_treatments = num_treatments

        self.limit_variance = limit_variance
        self.fix_variance = fix_variance
        self.limit_shape = limit_shape

        self.weight_init = weight_init
        if self.num_treatments !=2:
            raise NotImplementedError("X-learner only supports binary treatments")

        self.step_one_model = T_learner(
            network_name=self.network_name,
            input_dim=self.input_dim,
            hidden_sizes=self.hidden_sizes_outcome,
            outcome_dist=self.outcome_dist,
            num_treatments=self.num_treatments,
            experiment=f'{self.experiment}_step_one',
            model_path=self.model_path,
            dropout=self.dropout_outcome,
            activation=self.activation_outcome,
            device=self.device,
            save=self.save,
            limit_variance=self.limit_variance,
            fix_variance=self.fix_variance,
            limit_shape=self.limit_shape,
            optim_config=self.optim_config,
            wandblog=self.wandblog,
            weight_init=self.weight_init
        )

        self.step_two_model = T_learner(
            network_name=self.network_name,
            input_dim=input_dim,
            hidden_sizes=hidden_sizes_ite,
            outcome_dist=ite_dist,
            num_treatments=num_treatments,
            experiment=f'{experiment}_step_two',
            model_path=model_path,
            dropout=dropout_ite,
            activation=activation_ite,
            device=device,
            save=save,
            limit_variance=limit_variance,
            fix_variance=fix_variance,
            limit_shape=limit_shape,
            optim_config=optim_config,
            wandblog=wandblog,
            weight_init=weight_init
        )

        self.propensity_model = Treatment_Model(
            network_name = self.network_name,
            input_dim=input_dim,
            hidden_sizes=hidden_sizes_propensity,
            num_treatments=num_treatments,
            experiment=f'{experiment}_propensity',
            model_path=model_path,
            dropout=dropout_propensity,
            activation=activation_propensity,
            device=device,
            save=save,
            optim_config=optim_config,
            wandblog=wandblog,
            weight_init=weight_init
        )


    def forward(self, x_dict):
        # x = x_dict['x']

        y_dist_list = self.step_one_model(x_dict)
        propensity_score = self.propensity_model(x_dict)

        ite_pred_list = self.step_two_model(x_dict)
        if self.step_two_model.outcome_dist == 'not-specified':
            ite_pred_0, ite_pred_1 = ite_pred_list[0], ite_pred_list[1]
        else:
            ite_pred_0, ite_pred_1 = ite_pred_list[0].mean, ite_pred_list[1].mean
        ite_pred = ite_pred_0*propensity_score + ite_pred_1*(1-propensity_score)

        return {'y_dist_list': y_dist_list, 'ite_pred': ite_pred, 'ite_pred_list': ite_pred_list, 'propensity_score_pred': propensity_score}

    def fit(self, X_train, y_train, t_train=None, c_train=None,
            X_val=None, y_val=None, t_val=None, c_val=None,
            epochs_outcome=1000, epochs_ite=1000, epochs_propensity=1000, batch_size=256):

        # transform all inputs to tensors
        X_train = torch.tensor(X_train, dtype=torch.float).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float).to(self.device)
        t_train = torch.tensor(t_train, dtype=torch.float).to(self.device)
        if c_train is not None:
            c_train = torch.tensor(c_train, dtype=torch.float).to(self.device)
        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float).to(self.device)
            t_val = torch.tensor(t_val, dtype=torch.float).to(self.device)
            if c_val is not None:
                c_val = torch.tensor(c_val, dtype=torch.float).to(self.device)

        loss_step_one = self.step_one_model.fit(X_train, y_train, t_train, c_train, X_val, y_val, t_val, c_val, epochs_outcome, batch_size)
        loss_propensity = self.propensity_model.fit(X_train, y_train, t_train, c_train, X_val, y_val, t_val, c_val, epochs_propensity,
                                  batch_size)

        output_first_stage_train = self.step_one_model.predict(X_train)
        assert len(output_first_stage_train) == 2, 'Only binary treatments are supported'

        y1_pred_train = (output_first_stage_train['y_dist_list'][1]).mean
        y0_pred_train = (output_first_stage_train['y_dist_list'][0]).mean


        control_index_train = torch.tensor(t_train == 0, dtype=torch.bool)
        treated_index_train = torch.tensor(t_train == 1, dtype=torch.bool)

        y_train = y_train.view(-1,1)

        pred_ite_control_train = y1_pred_train[control_index_train] - y_train[control_index_train]
        pred_ite_treated_train = y_train[treated_index_train] - y0_pred_train[treated_index_train]

        y_ite_train = torch.zeros(len(y_train),1)
        y_ite_train[control_index_train] = pred_ite_control_train
        y_ite_train[treated_index_train] = pred_ite_treated_train

        # VALIDATION
        if X_val is not None:
            output_first_stage_val = self.step_one_model.predict(X_val)
            assert len(output_first_stage_val) == 2, 'Only binary treatments are supported'

            y1_pred_val = (output_first_stage_val['y_dist_list'][1]).mean
            y0_pred_val = (output_first_stage_val['y_dist_list'][0]).mean


            control_index_val = torch.tensor(t_val == 0, dtype=torch.bool)
            treated_index_val = torch.tensor(t_val == 1, dtype=torch.bool)

            y_val = y_val.view(-1,1)

            pred_ite_control_val = y1_pred_val[control_index_val] - y_val[control_index_val]
            pred_ite_treated_val = y_val[treated_index_val] - y0_pred_val[treated_index_val]

            y_ite_val = torch.zeros(len(y_val),1)
            y_ite_val[control_index_val] = pred_ite_control_val
            y_ite_val[treated_index_val] = pred_ite_treated_val
        else:
            y_ite_val = None

        loss_step_two = self.step_two_model.fit(X_train, y_ite_train, t_train, c_train, X_val, y_ite_val, t_val, c_val, epochs_ite,
                                batch_size)

        return {'loss_step_one': loss_step_one[-1], 'loss_step_two': loss_step_two[-1], 'loss_propensity': loss_propensity[-1]}

    @torch.no_grad()
    def predict(self, X_test, load_model='best'):
        self.load(load_model)

        self.step_one_model.eval()
        self.step_two_model.eval()
        self.propensity_model.eval()

        X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
        return self.forward({'x': X_test})

    def load(self, load_model='best'):
        self.step_one_model.load(load_model)
        self.step_two_model.load(load_model)
        self.propensity_model.load(load_model)


