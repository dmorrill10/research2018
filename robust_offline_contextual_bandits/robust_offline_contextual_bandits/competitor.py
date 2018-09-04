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


class TabularCompetitor(Competitor):
    @classmethod
    def new_rep(cls, x):
        return TabularRepresentationWithFixedInputs(x)

    def __init__(self, x, num_actions, policy_model_factory=RrmPolicyModel):
        rep = self.__class__.new_rep(x)
        policy_model = policy_model_factory(
            new_ff_policy_model(num_actions, rep.num_features()))
        super(TabularCompetitor, self).__init__(rep, policy_model,
                                                new_t_inv_gd_optimizer(1.0))


class TileCodingCompetitor(Competitor):
    @classmethod
    def new_rep(cls, num_tilings, num_tiles, x):
        return TileCodingRepresentationWithFixedInputs(num_tilings, num_tiles,
                                                       x)

    def __init__(self,
                 num_tilings,
                 num_tiles,
                 x,
                 num_actions,
                 policy_model_factory=RrmPolicyModel,
                 learning_rate_scale=1.0):
        rep = self.__class__.new_rep(num_tilings, num_tiles, x)
        policy_model = policy_model_factory(
            new_ff_policy_model(num_actions, rep.num_features()))
        super(TileCodingCompetitor, self).__init__(
            rep, policy_model,
            new_t_inv_gd_optimizer(learning_rate_scale * rep.learning_rate()))


class NnCompetitor(Competitor):
    def __init__(self,
                 network_factory,
                 x,
                 optimizer,
                 policy_model_factory=RrmPolicyModel):
        rep = self.__class__.new_rep(x)
        policy_model = policy_model_factory(
            network_factory(rep.num_features()))
        super(NnCompetitor, self).__init__(rep, policy_model, optimizer)


class RawNnCompetitor(NnCompetitor):
    @classmethod
    def new_rep(cls, x):
        return RawRepresentationWithFixedInputs(x)


class LiftAndProjectNnCompetitor(NnCompetitor):
    @classmethod
    def new_rep(cls, x):
        return LiftAndProjectRepresentationWithFixedInputs(x)
