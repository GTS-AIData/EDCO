from typing import Type
from codetiming import Timer

from mindspeed_llm.tasks.posttrain.rlxf.ray_trainer.ppo_trainer import ResourcePoolManager, Role
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.ray.megatron import NVMegatronRayWorkerGroup
from mindspeed_llm.tasks.posttrain.rlxf.training.core_algos import compute_grpo_data_metrics, reduce_metrics, \
    compute_advantage, compute_score, FixedKLController, AdaptiveKLController
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base import Worker
from mindspeed_llm.tasks.posttrain.rlxf.single_controller.ray.base import create_colocated_worker_cls, \
    set_actor_infer_world_size, set_actor_train_world_size, RayClassWithInitArgs
from mindspeed_llm.tasks.posttrain.rlxf.utils.loggers import Loggers
from mindspeed_llm.tasks.posttrain.rlxf.workers.actor_train_infer import PPOActorWorker
from mindspeed_llm.tasks.posttrain.rlxf.workers.reference import ReferenceWorker
from mindspeed_llm.tasks.posttrain.rlxf.workers.reward import RewardWorker


WorkerType = Type[Worker]


class RayGRPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(self, config):

        self.config = config
        if hasattr(self.config.training, "dataset_additional_keys"):
            self.config.training.dataset_additional_keys = config.training.dataset_additional_keys.strip().split(" ") if config.training.dataset_additional_keys else []

        self.role_worker_mapping = {
            Role.ActorRollout: PPOActorWorker,
            Role.RefPolicy: ReferenceWorker,
            Role.RewardModel: RewardWorker
        }
        actor_pool_id = 'actor_pool'
        ref_pool_id = 'ref_pool'
        reward_pool_id = 'reward_pool'

        if config.resource_pool.reward:
            resource_pool_spec = {
                actor_pool_id: config.resource_pool.actor_rollout,
                ref_pool_id: config.resource_pool.ref,
                reward_pool_id: config.resource_pool.reward,
            }
            mapping = {
                Role.ActorRollout: actor_pool_id,
                Role.RefPolicy: ref_pool_id,
                Role.RewardModel: reward_pool_id,
            }
        else:
            resource_pool_spec = {
                actor_pool_id: config.resource_pool.actor_rollout,
                ref_pool_id: config.resource_pool.ref
            }

            mapping = {
                Role.ActorRollout: actor_pool_id,
                Role.RefPolicy: ref_pool_id
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

        if self.config.resource_pool.reward:
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

        if self.config.resource_pool.reward:
            self.reward_wg = all_wg.get('reward')
            self.reward_wg.initialize()
        else:
            self.reward_wg = None

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

                with Timer(name='adv', logger=None) as timer:
                    # compute rm scores.
                    batch = compute_score(
                        self.reward_wg,
                        batch,
                        metrics,
                        self.config
                    )

                    # compute advantages, executed on the driver process
                    batch = compute_advantage(
                        batch,
                        self.config
                    )

                metrics['timing/adv'] = timer.last
                kl_info = {'kl_ctrl': self.kl_ctrl}
                batch.meta_info.update(kl_info)
                
                # update actor
                with Timer(name='update_actor', logger=None) as timer:
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                metrics['timing/update_actor'] = timer.last
                actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                metrics.update(actor_output_metrics)

                # collect metrics
                data_metrics = compute_grpo_data_metrics(batch=batch)
                metrics.update(data_metrics)
            metrics['timing/all'] = all_timer.last
            iteration += 1
            logger.info(metrics, iteration, self.config.training.train_iters)
            logger.flush()

            if iteration % self.config.training.save_interval == 0:
                self.actor_rollout_wg.save_checkpoint(iteration)
