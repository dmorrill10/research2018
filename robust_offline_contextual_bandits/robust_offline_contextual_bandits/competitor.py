from tf_contextual_prediction_with_expert_advice.rrm import \
    RrmPolicyModel
from robust_offline_contextual_bandits.representations import \
    TabularRepresentationWithFixedInputs, \
    RawRepresentationWithFixedInputs, \
    TileCodingRepresentationWithFixedInputs, \
    LiftAndProjectRepresentationWithFixedInputs
from robust_offline_contextual_bandits.policy import new_ff_policy_model
from robust_offline_contextual_bandits.optimizers import new_t_inv_gd_optimizer


class Competitor(object):
    def __init__(self, rep, policy_model, optimizer):
        self.rep = rep
        self.policy_model = policy_model
        self.optimizer = optimizer


class TabularCompetitor(object):
    def __init__(self, x, num_actions, policy_model_factory=RrmPolicyModel):
        rep = TabularRepresentationWithFixedInputs(x)
        policy_model = policy_model_factory(
            new_ff_policy_model(num_actions, rep.num_features()))
        super(TabularCompetitor, self).__init__(rep, policy_model,
                                                new_t_inv_gd_optimizer(1.0))


class TileCodingCompetitor(object):
    def __init__(self,
                 num_tilings,
                 num_tiles,
                 x,
                 num_actions,
                 policy_model_factory=RrmPolicyModel,
                 learning_rate_scale=1.0):
        rep = TileCodingRepresentationWithFixedInputs(num_tilings, num_tiles,
                                                      x)
        policy_model = policy_model_factory(
            new_ff_policy_model(num_actions, rep.num_features()))
        super(TileCodingCompetitor, self).__init__(
            rep, policy_model,
            new_t_inv_gd_optimizer(learning_rate_scale * rep.learning_rate()))


class RawNnCompetitor(object):
    def __init__(self,
                 num_units,
                 x,
                 num_actions,
                 optimizer,
                 policy_model_factory=RrmPolicyModel):
        rep = RawRepresentationWithFixedInputs(x)
        policy_model = policy_model_factory(
            new_ff_policy_model(
                num_actions,
                rep.num_features(),
                num_hidden=1,
                num_units1=num_units))
        super(RawNnCompetitor, self).__init__(rep, policy_model, optimizer)


class LiftAndProjectNnCompetitor(object):
    def __init__(self,
                 num_units,
                 x,
                 num_actions,
                 optimizer,
                 policy_model_factory=RrmPolicyModel):
        rep = LiftAndProjectRepresentationWithFixedInputs(x)
        policy_model = policy_model_factory(
            new_ff_policy_model(
                num_actions,
                rep.num_features(),
                num_hidden=1,
                num_units1=num_units))
        super(LiftAndProjectNnCompetitor, self).__init__(
            rep, policy_model, optimizer)
