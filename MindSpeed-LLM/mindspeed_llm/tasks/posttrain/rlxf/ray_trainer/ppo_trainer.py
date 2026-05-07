from dataclasses import dataclass, field
from enum import Enum
from typing import Type, Dict, List

from codetiming import Timer

from mindspeed_llm.tasks.posttrain.rlxf.single_controller.ray.megatron import NVMegatronRayWorkerGroup
from mindspeed_llm.tasks.posttrain.rlxf.training.core_algos import compute_data_metrics, reduce_metrics, compute_advantage, \
    apply_kl_penalty, FixedKLController, AdaptiveKLController
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base import Worker
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.ray.base import create_colocated_worker_cls, \
    set_actor_infer_world_size, set_actor_train_world_size, RayResourcePool, RayClassWithInitArgs
from mindspeed_llm.tasks.posttrain.rlxf.utils.loggers import Loggers
from mindspeed_llm.tasks.posttrain.rlxf.workers.critic import CriticWorker
from mindspeed_llm.tasks.posttrain.rlxf.workers.actor_train_infer import PPOActorWorker
from mindspeed_llm.tasks.posttrain.rlxf.workers.reference import ReferenceWorker
from mindspeed_llm.tasks.posttrain.rlxf.workers.reward import RewardWorker

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    ActorRollout = 0
    Critic = 1
    RefPolicy = 2
    RewardModel = 3


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: Dict[str, List[int]]
    mapping: Dict[Role, str]
    resource_pool_dict: Dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        print(f"role:{role}, self.mapping[role]:{self.mapping[role]}")
        return self.resource_pool_dict[self.mapping[role]]


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self, config):

        self.config = config
        self.role_worker_mapping = {
            Role.ActorRollout: PPOActorWorker,
            Role.RefPolicy: ReferenceWorker,
            Role.Critic: CriticWorker,
            Role.RewardModel: RewardWorker
        }
        actor_pool_id = 'actor_pool'
        ref_pool_id = 'ref_pool'
        critic_pool_id = 'critic_pool'
        reward_pool_id = 'reward_pool'

        resource_pool_spec = {
            actor_pool_id: config.resource_pool.actor_rollout,
            ref_pool_id: config.resource_pool.ref,
            critic_pool_id: config.resource_pool.critic,
            reward_pool_id: config.resource_pool.reward,
        }

        mapping = {
            Role.ActorRollout: actor_pool_id,
            Role.RefPolicy: ref_pool_id,
            Role.Critic: critic_pool_id,
            Role.RewardModel: reward_pool_id,
        }

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
        self.use_reference_policy = Role.RefPolicy in self.role_worker_mapping
        self.ray_worker_group_cls = NVMegatronRayWorkerGroup

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                if config.algorithm.kl_ctrl.horizon <= 0:
                    raise ValueError(f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}')
                self.kl_ctrl = AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                    target_kl=config.algorithm.kl_ctrl.target_kl,
                                                    horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = FixedKLController(kl_coef=0.)
        self.init_workers()

    def init_workers(self):
        """Init resource pool and worker group"""
        set_actor_infer_world_size(self.config.actor_rollout_ref.actor_rollout.num_gpus_for_infer)
        set_actor_train_world_size(self.config.actor_rollout_ref.actor_rollout.num_gpus_for_train)
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                 config=self.config,
                                                 role='actor_rollout')
        self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
        ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                              config=self.config,
                                              role='ref')
        self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
        critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic],
                                          config=self.config,
                                          role='critic')
        self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
        reward_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.RewardModel],
                                          config=self.config,
                                          role='reward')
        self.resource_pool_to_cls[resource_pool]['reward'] = reward_cls

        # initialize WorkerGroup
        all_wg = {}
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.ref_policy_wg = all_wg.get('ref')
        self.ref_policy_wg.initialize()

        self.actor_rollout_wg = all_wg.get('actor_rollout')
        self.actor_rollout_wg.initialize()

        self.critic_wg = all_wg.get('critic')
        self.critic_wg.initialize()

        self.reward_wg = all_wg.get('reward')
        self.reward_wg.initialize()

    def train(self):
        """
        The training loop of PPO.
        """
        logger = Loggers()

        iteration = self.actor_rollout_wg.get_iteration()[0]

        while iteration < self.config.training.train_iters:
            with Timer(name='gen', logger=None) as all_timer:
                metrics = {}
                self.actor_rollout_wg.auto_mapping()
                # generate a batch
                with Timer(name='gen', logger=None) as timer:
                    batch = self.actor_rollout_wg.generate_sequences()
                    batch = self.actor_rollout_wg.get_log_probs(batch)
                metrics['timing/gen'] = timer.last

                if self.use_reference_policy:
                    # compute reference log_prob
                    with Timer(name='ref', logger=None) as timer:
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                    metrics['timing/ref'] = timer.last

                # compute values
                with Timer(name='values', logger=None) as timer:
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)
                metrics['timing/values'] = timer.last

                with Timer(name='adv', logger=None) as timer:
                    # compute rm scores.
                    reward_tensor = self.reward_wg.compute_rm_score(batch)
                    batch = batch.union(reward_tensor)

                    # compute rewards. apply_kl_penalty if available
                    batch, kl_metrics = apply_kl_penalty(self.config, batch,
                                                         kl_ctrl=self.kl_ctrl)
                    metrics.update(kl_metrics)

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        self.config
                    )
                metrics['timing/adv'] = timer.last

                # update critic
                with Timer(name='update_critic', logger=None) as timer:
                    critic_output = self.critic_wg.update_critic(batch)
                metrics['timing/update_critic'] = timer.last
                critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                metrics.update(critic_output_metrics)

                # update actor
                with Timer(name='update_actor', logger=None) as timer:
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics['timing/update_actor'] = timer.last
                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                metrics.update(actor_output_metrics)

                # collect metrics
                data_metrics = compute_data_metrics(batch=batch)
                metrics.update(data_metrics)
            metrics['timing/all'] = all_timer.last
            iteration += 1
            logger.info(metrics, iteration, self.config.training.train_iters)

            if iteration % self.config.training.save_interval == 0:
                self.critic_wg.save_checkpoint(iteration)
                self.actor_rollout_wg.save_checkpoint(iteration)

