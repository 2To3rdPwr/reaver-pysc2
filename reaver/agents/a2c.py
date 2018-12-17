import gin.tf
import tensorflow as tf

from reaver.utils import Logger
from reaver.envs.base import Spec
from reaver.utils.tensorflow import SessionManager
from reaver.utils.typing import ModelBuilder, PolicyType
from reaver.agents.base import SyncRunningAgent, ActorCriticAgent


@gin.configurable
class AdvantageActorCriticAgent(SyncRunningAgent, ActorCriticAgent):
    """
    A2C: a synchronous version of Asynchronous Advantage Actor Critic (A3C)
    See article for more details: https://arxiv.org/abs/1602.01783
    """
    def __init__(
        self,
        obs_spec: Spec,
        act_spec: Spec,
        model_fn: ModelBuilder,
        policy_cls: PolicyType,
        sess_mgr: SessionManager = None,
        n_envs=4,
        traj_len=16,
        batch_sz=16,
        discount=0.99,
        gae_lambda=0.95,
        clip_rewards=0.0,
        normalize_advantages=True,
        clip_grads_norm=0.0,
        value_coef=0.5,
        entropy_coef=0.001,
        optimizer=tf.train.AdamOptimizer(),
        logger=Logger(),
    ):
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        SyncRunningAgent.__init__(self, n_envs)
        ActorCriticAgent.__init__(
            self, obs_spec, act_spec, model_fn, policy_cls, sess_mgr, traj_len, batch_sz, discount,
            gae_lambda, clip_rewards, normalize_advantages, clip_grads_norm, optimizer, logger
        )

    def loss_fn(self):
        adv = tf.placeholder(tf.float32, [None], name="advantages")
        returns = tf.placeholder(tf.float32, [None], name="returns")

        policy_loss = -tf.reduce_mean(self.policy.logli * adv)
        value_loss = tf.reduce_mean((self.value - returns)**2) * self.value_coef
        entropy_loss = tf.reduce_mean(self.policy.entropy) * self.entropy_coef
        # we want to reduce policy and value errors, and maximize entropy
        # but since optimizer is minimizing the signs are opposite
        full_loss = policy_loss + value_loss - entropy_loss

        try:
            with open("loss_fn.txt", "x+") as f:
                f.write("out\n")
                f.write("full_loss: {0} type: {1}\n".format(type(full_loss), full_loss.dtype))
                f.write("policy_loss: {0} type: {1}\n".format(type(policy_loss), policy_loss.dtype))
                f.write("value_loss: {0} type: {1}\n".format(type(value_loss), value_loss.dtype))
                f.write("entropy_loss: {0} type: {1}\n".format(type(entropy_loss), entropy_loss.dtype))
                f.write("adv: {0} type: {1}\n".format(type(adv), adv.dtype))
                f.write("returns: {0} type: {1}\n".format(type(returns), returns.dtype))
                f.close()
        except FileExistsError:
            print("")

        return full_loss, [policy_loss, value_loss, entropy_loss], [adv, returns]
