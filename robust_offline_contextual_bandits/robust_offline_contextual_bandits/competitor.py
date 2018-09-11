from tf_contextual_prediction_with_expert_advice.rrm import \
    RrmPolicyModel
from robust_offline_contextual_bandits.representations import \
    RepresentationWithFixedInputs
from robust_offline_contextual_bandits.policy import new_ff_policy_model
from robust_offline_contextual_bandits.optimizers import new_t_inv_gd_optimizer


class Competitor(object):
    @classmethod
    def from_network_factory(cls,
                             rep,
                             network_factory,
                             optimizer,
                             policy_model_factory=RrmPolicyModel):
        policy_model = policy_model_factory(
            network_factory(rep.num_features()))
        return cls(rep, policy_model, optimizer)

    @classmethod
    def tabular(cls, x, num_actions, **kwargs):
        rep = RepresentationWithFixedInputs.tabular(x)
        optimizer = new_t_inv_gd_optimizer(rep.learning_rate())
        return cls.from_network_factory(
            rep, lambda nf: new_ff_policy_model(num_actions, nf), optimizer,
            **kwargs)

    @classmethod
    def tile_coding(cls,
                    x,
                    num_actions,
                    num_tiling_pairs,
                    policy_model_factory=RrmPolicyModel,
                    **kwargs):
        rep = RepresentationWithFixedInputs.dense_tile_coding(
            x, num_tiling_pairs, **kwargs)
        optimizer = new_t_inv_gd_optimizer(rep.learning_rate())
        return cls.from_network_factory(
            rep,
            lambda nf: new_ff_policy_model(num_actions, nf),
            optimizer,
            policy_model_factory=RrmPolicyModel)

    def __init__(self, rep, policy_model, optimizer):
        self.rep = rep
        self.policy_model = policy_model
        self.optimizer = optimizer
