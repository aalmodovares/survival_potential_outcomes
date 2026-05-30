import torch
from patsy.contrasts import Treatment
from torch import nn
from itertools import chain

from models.wrapper import Wrapper
from models.networks.regressors import outcome_regressor, treatment_regressor
from models.networks.encoder import Encoder
from models.networks.decoder import Decoder

from models.potential_outcomes.t_learner import T_learner
from models.potential_outcomes.tarnet import TARNet
from models.potential_outcomes.regressors_models import Treatment_Model, Outcome_Model

from torch.distributions import Independent, Normal

# Adapter from Zhang, W., Liu, L., & Li, J. (2021, May). Treatment effect estimation with disentangled latent factors.
# In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 35, No. 12, pp. 10923-10930).

class TEDVAE(Wrapper):
    def __init__(self, input_dim,
                 encoder_name, decoder_name, treatment_regressor_name, outcome_regressor_name,
                 latent_dim_t, latent_dim_c, latent_dim_y,
                 hidden_sizes_encoder, hidden_sizes_decoder, hidden_sizes_treatment_regressor, hidden_sizes_outcome_regressor,
                 outcome_dist, num_treatments,
                 latent_dist, data_types,
                 loss_weights,
                 experiment, model_path,
                 dropout,
                 activation_encoder, activation_decoder, activation_treatment_regressor, activation_outcome_regressor,
                 device,
                 weight_init_encoder=None, weight_init_decoder=None, weight_init_treatment_regressor=None, weight_init_outcome_regressor=None,
                 save=True,
                 limit_variance_encoder=False, limit_variance_decoder=False,
                 limit_variance_outcome=False, min_std_outcome=None, max_std_outcome=None,
                 fix_variance_encoder=False, fix_variance_decoder=False, fix_variance_outcome=False, fixed_std_outcome=None,
                 limit_shape_encoder=False, limit_shape_decoder=False,
                 limit_shape_outcome=False, min_shape_outcome=None, max_shape_outcome=None,
                 limit_scale_outcome=False, min_scale_outcome=None, max_scale_outcome=None,
                 # censoring model, optional: depending on the values provided, automatically chooses between model M1 and M2 from the paper
                 censoring_model_name=None,
                 hidden_sizes_censoring_model=None, activation_censoring_model=None, weight_init_censoring_model=None,
                 censoring_dist=None,
                 limit_variance_censoring=False, min_std_censoring=None, max_std_censoring=None,
                 fix_variance_censoring=False, fixed_std_censoring=None,
                 limit_shape_censoring=False, min_shape_censoring=None, max_shape_censoring=None,
                 limit_scale_censoring=False, min_scale_censoring=None, max_scale_censoring=None,
                 optim_config=None, wandblog=False, disable_tqdm=False):

        super(TEDVAE, self).__init__('tedvae', device, experiment, model_path, save, optim_config, wandblog, disable_tqdm)

        self.input_dim = input_dim
        self.latent_dim_t = latent_dim_t
        self.latent_dim_c = latent_dim_c
        self.latent_dim_y = latent_dim_y
        self.hidden_sizes_encoder = hidden_sizes_encoder
        self.hidden_sizes_decoder = hidden_sizes_decoder
        self.hidden_sizes_treatment_regressor = hidden_sizes_treatment_regressor
        self.hidden_sizes_outcome_regressor = hidden_sizes_outcome_regressor
        self.outcome_dist = outcome_dist
        self.num_treatments = num_treatments
        self.latent_dist = latent_dist
        self.data_types = data_types
        self.loss_weights = loss_weights
        self.dropout = dropout
        self.activation_encoder = activation_encoder
        self.activation_decoder = activation_decoder
        self.activation_treatment_regressor = activation_treatment_regressor
        self.activation_outcome_regressor = activation_outcome_regressor

        self.weight_init_encoder = weight_init_encoder
        self.weight_init_decoder = weight_init_decoder
        self.weight_init_treatment_regressor = weight_init_treatment_regressor
        self.weight_init_outcome_regressor = weight_init_outcome_regressor

        self.limit_variance_encoder = limit_variance_encoder
        self.limit_variance_decoder = limit_variance_decoder
        self.limit_variance_outcome = limit_variance_outcome
        self.min_std_outcome = min_std_outcome
        self.max_std_outcome = max_std_outcome
        self.fix_variance_encoder = fix_variance_encoder
        self.fix_variance_decoder = fix_variance_decoder
        self.fix_variance_outcome = fix_variance_outcome
        self.fixed_std_outcome = fixed_std_outcome
        self.limit_shape_encoder = limit_shape_encoder
        self.limit_shape_decoder = limit_shape_decoder
        self.limit_shape_outcome = limit_shape_outcome
        self.min_shape_outcome = min_shape_outcome
        self.max_shape_outcome = max_shape_outcome
        self.limit_scale_outcome = limit_scale_outcome
        self.min_scale_outcome = min_scale_outcome
        self.max_scale_outcome = max_scale_outcome

        self.latent_dim_all = self.latent_dim_t + self.latent_dim_c + self.latent_dim_y

        if self.loss_weights is None:
            self.loss_weights = {
                'w_disentanglement_t': 100,
                'w_disentanglement_y': 100,
                'w_disentanglement_c': 100,
                'w_kl': 1,
                'w_reconstruction': 1,
                'w_model_t': 1,
                'w_model_y': 1,
                'w_model_c': 1
            }

        self.encoder_t = Encoder(network_name=encoder_name,
                               input_dim=self.input_dim,
                               hidden_sizes=self.hidden_sizes_encoder,
                               latent_dim = self.latent_dim_t,
                               latent_dist=self.latent_dist,
                               dropout=self.dropout,
                               activation=self.activation_encoder,
                               device=self.device,
                               limit_variance=self.limit_variance_encoder,
                               fix_variance=self.fix_variance_encoder,
                               weight_init=self.weight_init_encoder)

        self.encoder_c = Encoder(network_name=encoder_name,
                               input_dim=self.input_dim,
                               hidden_sizes=self.hidden_sizes_encoder,
                               latent_dim = self.latent_dim_c,
                               latent_dist=self.latent_dist,
                               dropout=self.dropout,
                               activation=self.activation_encoder,
                               device=self.device,
                               limit_variance=self.limit_variance_encoder,
                               fix_variance=self.fix_variance_encoder,
                               weight_init=self.weight_init_encoder)

        self.encoder_y = Encoder(network_name=encoder_name,
                               input_dim=self.input_dim,
                               hidden_sizes=self.hidden_sizes_encoder,
                               latent_dim = self.latent_dim_y,
                               latent_dist=self.latent_dist,
                               dropout=self.dropout,
                               activation=self.activation_encoder,
                               device=self.device,
                               limit_variance=self.limit_variance_encoder,
                               fix_variance=self.fix_variance_encoder,
                               weight_init=self.weight_init_encoder)

        self.decoder = Decoder(network_name=decoder_name,
                               input_dim=self.latent_dim_all,
                               hidden_sizes=self.hidden_sizes_decoder,
                               data_types = self.data_types,
                               dropout=self.dropout,
                               activation=self.activation_decoder,
                               device=self.device,
                               limit_variance=self.limit_variance_decoder,
                               fix_variance=self.fix_variance_decoder,
                               limit_shape=self.limit_shape_decoder,
                               weight_init=self.weight_init_decoder)

        # There are two treatment regressors, one for the guide and one for the model
        # From TEDVAE paper
        self.treatment_guide = Treatment_Model(network_name=treatment_regressor_name,
                                                      input_dim=self.latent_dim_t + self.latent_dim_c,
                                                      hidden_sizes=self.hidden_sizes_treatment_regressor,
                                                      num_treatments=self.num_treatments,
                                                      dropout=self.dropout,
                                                      activation=self.activation_treatment_regressor,
                                                      device=self.device,
                                                      weight_init=self.weight_init_treatment_regressor,
                                                      experiment=self.experiment,
                                                      model_path=self.model_path,
                                                      is_subnetwork=True
                                               )
        self.treatment_model = Treatment_Model(network_name=treatment_regressor_name,
                                                      input_dim=self.latent_dim_t + self.latent_dim_c,
                                                      hidden_sizes=self.hidden_sizes_treatment_regressor,
                                                      num_treatments=self.num_treatments,
                                                      dropout=self.dropout,
                                                      activation=self.activation_outcome_regressor,
                                                      device=self.device,
                                                      weight_init=self.weight_init_treatment_regressor,
                                                      experiment=self.experiment,
                                                      model_path=self.model_path,
                                                      is_subnetwork=True
                                                   )


        self.outcome_model = T_learner(network_name = outcome_regressor_name,
                                       input_dim = self.latent_dim_y + self.latent_dim_c,
                                       hidden_sizes = self.hidden_sizes_outcome_regressor,
                                       outcome_dist = self.outcome_dist,
                                       num_treatments = self.num_treatments,
                                       experiment = self.experiment,
                                       model_path = self.model_path,
                                       dropout = self.dropout,
                                       activation = self.activation_outcome_regressor,
                                       device = self.device,
                                       save = self.save,
                                       limit_variance = self.limit_variance_outcome,
                                       min_std = self.min_std_outcome,
                                       max_std = self.max_std_outcome,
                                       fix_variance = self.fix_variance_outcome,
                                       fixed_std = self.fixed_std_outcome,
                                       limit_scale=self.limit_scale_outcome,
                                       min_scale=self.min_scale_outcome,
                                       max_scale=self.max_scale_outcome,
                                       limit_shape=self.limit_shape_outcome,
                                       min_shape=self.min_shape_outcome,
                                       max_shape=self.max_shape_outcome,
                                       optim_config = self.optim_config,
                                       wandblog = self.wandblog,
                                       weight_init=self.weight_init_outcome_regressor,
                                       is_subnetwork=True)

        self.outcome_guide = TARNet(network_name = outcome_regressor_name,
                                    input_dim = self.latent_dim_y + self.latent_dim_c,
                                    hidden_sizes_phi = self.hidden_sizes_outcome_regressor[:-1],
                                    hidden_sizes_y = [],                                          # todo: hardcoded to 0
                                    phi_dim = self.hidden_sizes_outcome_regressor[-1],
                                    outcome_dist = self.outcome_dist,
                                    num_treatments = self.num_treatments,
                                    experiment = self.experiment,
                                    model_path = self.model_path,
                                    dropout = self.dropout,
                                    activation = self.activation_outcome_regressor,
                                    device = self.device,
                                    save = self.save,
                                    limit_variance = self.limit_variance_outcome,
                                    min_std=self.min_std_outcome,
                                    max_std=self.max_std_outcome,
                                    fix_variance = self.fix_variance_outcome,
                                    fixed_std = self.fixed_std_outcome,
                                    limit_scale=self.limit_scale_outcome,
                                    min_scale=self.min_scale_outcome,
                                    max_scale=self.max_scale_outcome,
                                    limit_shape=self.limit_shape_outcome,
                                    min_shape=self.min_shape_outcome,
                                    max_shape=self.max_shape_outcome,
                                    optim_config = self.optim_config,
                                    wandblog = self.wandblog,
                                    weight_init_phi=self.weight_init_outcome_regressor,
                                    weight_init_y=self.weight_init_outcome_regressor,
                                    is_subnetwork=True)

        # Depenging on censoring, we may have M1 or M2
        self.censor = None
        if censoring_model_name is not None:
            self.censoring_model_name = censoring_model_name
            self.hidden_sizes_censoring_model = hidden_sizes_censoring_model
            self.activation_censoring_model = activation_censoring_model
            self.weight_init_censoring_model = weight_init_censoring_model
            self.censoring_dist = censoring_dist
            self.limit_variance_censoring = limit_variance_censoring
            self.fix_variance_censoring = fix_variance_censoring
            self.fixed_std_censoring = fixed_std_censoring
            self.limit_shape_censoring = limit_shape_censoring
            self.min_shape_censoring = min_shape_censoring
            self.max_shape_censoring = max_shape_censoring
            self.limit_scale_censoring = limit_scale_censoring
            self.min_scale_censoring = min_scale_censoring
            self.max_scale_censoring = max_scale_censoring

            if self.censoring_dist == 'binary' or self.censoring_dist == 'bernoulli':
                self.censor = 'M1'  # Binary prediction of censoring
                print('Using model M1 for censoring, which predicts binary censoring indicator')
            elif censoring_dist == 'weibull':
                self.censor = 'M2'  # Time to censoring prediction with Weibull distribution
                print('Using model M2 for censoring, which predicts time to censoring with a Weibull distribution')
            else:
                raise ValueError('censoring_dist must be either None or "weibull"')

        if self.censor is not None:
            self.censoring_model = T_learner(network_name=censoring_model_name,
                                             input_dim=self.latent_dim_t + self.latent_dim_c,
                                             hidden_sizes=self.hidden_sizes_censoring_model,
                                             outcome_dist=self.censoring_dist, # M1 or M2 depending on the censoring_dist provided
                                             num_treatments=self.num_treatments,
                                             experiment=self.experiment,
                                             model_path=self.model_path,
                                             dropout=self.dropout,
                                             activation=self.activation_censoring_model,
                                             device=self.device,
                                             save=self.save,
                                             limit_variance=self.limit_variance_censoring,
                                             fix_variance=self.fix_variance_censoring,
                                             fixed_std=self.fixed_std_censoring,
                                             limit_scale=self.limit_scale_censoring,
                                             min_scale=self.min_scale_censoring,
                                             max_scale=self.max_scale_censoring,
                                             limit_shape=self.limit_shape_censoring,
                                             min_shape=self.min_shape_censoring,
                                             max_shape=self.max_shape_censoring,
                                             optim_config=self.optim_config,
                                             wandblog=self.wandblog,
                                             weight_init=self.weight_init_censoring_model,
                                             is_subnetwork=True)

            self.censoring_guide = TARNet(network_name=censoring_model_name,
                                          input_dim=self.latent_dim_t + self.latent_dim_c,
                                          hidden_sizes_phi=self.hidden_sizes_censoring_model[:-1],
                                          hidden_sizes_y=[],  # todo: hardcoded to 0
                                          phi_dim=self.hidden_sizes_censoring_model[-1],
                                          outcome_dist=self.censoring_dist,
                                          num_treatments=self.num_treatments,
                                          experiment=self.experiment,
                                          model_path=self.model_path,
                                          dropout=self.dropout,
                                          activation=self.activation_censoring_model,
                                          device=self.device,
                                          save=self.save,
                                          limit_variance=self.limit_variance_censoring,
                                          fix_variance=self.fix_variance_censoring,
                                          fixed_std=self.fixed_std_censoring,
                                          limit_scale=self.limit_scale_censoring,
                                          min_scale=self.min_scale_censoring,
                                          max_scale=self.max_scale_censoring,
                                          limit_shape=self.limit_shape_censoring,
                                          min_shape=self.min_shape_censoring,
                                          max_shape=self.max_shape_censoring,
                                          optim_config=self.optim_config,
                                          wandblog=self.wandblog,
                                          weight_init_phi=self.weight_init_censoring_model,
                                          weight_init_y=self.weight_init_censoring_model,
                                          is_subnetwork=True)

        self.prior_t = Independent(Normal(loc=torch.zeros(self.latent_dim_t), scale=torch.ones(self.latent_dim_t)), 1)
        self.prior_c = Independent(Normal(loc=torch.zeros(self.latent_dim_c), scale=torch.ones(self.latent_dim_c)), 1)
        self.prior_y = Independent(Normal(loc=torch.zeros(self.latent_dim_y), scale=torch.ones(self.latent_dim_y)), 1)

        self.get_optim()

    def forward(self, x_dict):

        x = x_dict['x']

        # Encode
        q_z_t = self.encoder_t(x)
        q_z_c = self.encoder_c(x)
        q_z_y = self.encoder_y(x)

        z_t = q_z_t.rsample()
        z_c = q_z_c.rsample()
        z_y = q_z_y.rsample()

        z_tcy = torch.cat([z_t, z_c, z_y], dim=1)
        z_tc = torch.cat([z_t, z_c], dim=1)
        z_cy = torch.cat([z_c, z_y], dim=1)

        # Decode
        p_x_given_z = self.decoder(z_tcy)

        # Treatment
        q_t_guide = self.treatment_guide({'x':z_tc})
        p_t_model = self.treatment_model({'x': z_tc})

        # Outcome
        q_y_quide_list = self.outcome_guide({'x': z_cy})
        p_y_model_list = self.outcome_model({'x': z_cy})

        output_dict = {'q_z_t': q_z_t, 'q_z_c': q_z_c, 'q_z_y': q_z_y,
                       'z_t': z_t, 'z_c': z_c, 'z_y': z_y,
                       'p_x_given_z': p_x_given_z,
                       'q_t_guide': q_t_guide, 'p_t_model': p_t_model,
                       'q_y_guide_list': q_y_quide_list, 'p_y_model_list': p_y_model_list}

        if self.censor is not None:
            q_c_guide_list = self.censoring_guide({'x': z_tc})
            p_c_model_list = self.censoring_model({'x': z_tc})
            output_dict['q_c_guide_list'] = q_c_guide_list
            output_dict['p_c_model_list'] = p_c_model_list

        return output_dict


    def loss(self, output, true_values):
        x_true = true_values['x']
        y_true = true_values['y']
        t_true = true_values['t']
        c_true = true_values['c']
        mask_true = true_values['mask'] if 'mask' in true_values else None

        z_t = output['z_t']
        z_c = output['z_c']
        z_y = output['z_y']

        q_z_t = output['q_z_t']
        q_z_c = output['q_z_c']
        q_z_y = output['q_z_y']

        p_x_given_z = output['p_x_given_z']

        q_t_guide = output['q_t_guide']
        p_t_model = output['p_t_model']

        q_y_guide_list = output['q_y_guide_list']
        p_y_model_list = output['p_y_model_list']

        assert t_true is not None, 'treatment is None'
        # Reconstruction loss
        reconstruction_loss = self.decoder.reconstruction_log_prob(p_x_given_z, x_true, mask_true) # batch size x num_features
        reconstruction_loss = - torch.sum(reconstruction_loss, dim=-1) # sum across all features, batch_size remains

        # KL divergence computed with MonteCarlo to avoid variance reduction
        log_q_z_t = q_z_t.log_prob(z_t) #batch size x latent_dim_t // log q(z_t|x)
        log_q_z_c = q_z_c.log_prob(z_c) #batch size x latent_dim_t // log q(z_c|x)
        log_q_z_y = q_z_y.log_prob(z_y) #batch size x latent_dim_t // log q(z_y|x)

        log_p_z_t = self.prior_t.log_prob(z_t) #batch size, already summed  // log p(z_t)
        log_p_z_c = self.prior_c.log_prob(z_c) #batch size, already summed  // log p(z_c)
        log_p_z_y = self.prior_y.log_prob(z_y) #batch size, already summed  // log p(z_y)

        kl_t = log_q_z_t - log_p_z_t # batch size x latent_dim_t
        kl_c = log_q_z_c - log_p_z_c # batch size x latent_dim_c
        kl_y = log_q_z_y - log_p_z_y # batch size x latent_dim_y

        # treatment loss
        log_prob_q_t = q_t_guide.log_prob(t_true) # batch size
        log_prob_p_t = p_t_model.log_prob(t_true) # batch size

        # outcome loss
        loss_q_y = self.outcome_guide.loss(q_y_guide_list, {'y': y_true, 'c': c_true, 't': t_true})['loss'] # this is already averaged
        loss_p_y = self.outcome_model.loss(p_y_model_list, {'y': y_true, 'c': c_true, 't': t_true})['loss'] # this is already averaged

        loss_disentanglement = - self.loss_weights['w_disentanglement_t'] * log_prob_q_t.mean() + self.loss_weights['w_disentanglement_y'] * loss_q_y
        loss_reconstruction = self.loss_weights['w_reconstruction'] * reconstruction_loss.mean()
        loss_kl = self.loss_weights['w_kl'] * (kl_t.mean() + kl_c.mean() + kl_y.mean())
        loss_model = - self.loss_weights['w_model_t'] * log_prob_p_t.mean() + self.loss_weights['w_model_y'] * loss_p_y

        elbo = -(loss_reconstruction + loss_kl)

        loss = loss_disentanglement + loss_reconstruction + loss_kl + loss_model

        if self.censor is not None:
            q_c_guide_list = output['q_c_guide_list']
            p_c_model_list = output['p_c_model_list']
            if self.censor == 'M1': # The censoring model predicts censoring indicator
                loss_q_c = self.censoring_guide.loss(q_c_guide_list, {'y': c_true, 'c': None, 't': t_true})['loss']  # this is already averaged, note that we now are trying to estimate the censoring time, so our event is censoring!
                loss_p_c = self.censoring_model.loss(p_c_model_list, {'y': c_true, 'c': None, 't': t_true})['loss']  # this is already averaged
            if self.censor == 'M2': # The censoring model predicts time to censoring
                loss_q_c = self.censoring_guide.loss(q_c_guide_list, {'y': y_true, 'c': 1 - c_true, 't': t_true})['loss']  # this is already averaged, note that we now are trying to estimate the censoring time, so our event is censoring!
                loss_p_c = self.censoring_model.loss(p_c_model_list, {'y': y_true, 'c': 1 - c_true, 't': t_true})['loss']  # this is already averaged
            loss += self.loss_weights['w_model_c'] * loss_p_c
            loss_disentanglement += self.loss_weights['w_disentanglement_c'] * loss_q_c


        loss_dict = {'loss': loss.mean(),
                     'reconstruction_loss': reconstruction_loss.mean(),
                     'log_q_z_t': log_q_z_t.mean(),
                     'log_q_z_c': log_q_z_c.mean(),
                     'log_q_z_y': log_q_z_y.mean(),
                     'log_p_z_t': log_p_z_t.mean(),
                     'log_p_z_c': log_p_z_c.mean(),
                     'log_p_z_y': log_p_z_y.mean(),
                     'kl_t': kl_t.mean(),
                     'kl_c': kl_c.mean(),
                     'kl_y': kl_y.mean(),
                     'kl_total': kl_t.mean() + kl_c.mean() + kl_y.mean(),
                     'log_prob_q_t': log_prob_q_t.mean(),
                     'log_prob_p_t': log_prob_p_t.mean(),
                     'loss_q_y': loss_q_y,
                     'loss_p_y': loss_p_y,
                     'loss_model': loss_model,
                     'loss_disentanglement': loss_disentanglement,
                     '-elbo': -elbo}
        #if self.censor == 'M1':
        #    loss_dict['loss_p_censoring'] = loss_p_censoring
        if self.censor is not None:
            loss_dict['loss_q_c'] = loss_q_c
            loss_dict['loss_p_c'] = loss_p_c

        return loss_dict

    @torch.no_grad()
    def predict(self, X_test, load_model:str | None  = 'best'):
        if load_model is not None:
            self.load(load_model)
        self.eval()
        X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)

        z_t = self.encoder_t(X_test).sample()
        z_c = self.encoder_c(X_test).sample()
        z_y = self.encoder_y(X_test).sample()

        z_tc = torch.cat([z_t, z_c], dim=1)
        z_cy = torch.cat([z_c, z_y], dim=1)

        y_dist_list = self.outcome_model({'x': z_cy})

        t_dist = self.treatment_model({'x': z_tc})
        propensity_score_pred = t_dist.mean

        output_dict = {'propensity_score_pred': propensity_score_pred, 'y_dist_list': y_dist_list}

        if self.censor is not None:
            c_dist_list = self.censoring_model({'x': z_tc})
            output_dict['c_dist_list'] = c_dist_list

        return output_dict


