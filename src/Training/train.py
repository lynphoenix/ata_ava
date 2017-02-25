import image_processing
import DistributedConfig
import BasicConfig
from LearningModule import LearningModule, StartTraining

import tensorflow as tf
import argparse

def ata_log(log_str):
    tf.logging.info(log_str)


def build_cluster_spec(args):

    ps_hosts = args.ps_hosts.split(',')
    worker_hosts = args.worker_hosts.split(',')

    cluster_spec = tf.train.ClusterSpec({'ps': ps_hosts, 'worker': worker_hosts})

    server = tf.train.Server(
        {'ps': ps_hosts, 'worker': worker_hosts},
        job_name=args.job_name, task_index=args.task_id)


def DistributedConfig(args, cluster_spec):
    num_workers = len(cluster_spec.as_dict()['worker'])
    num_parameter_servers = len(cluster_spec.as_dict()['ps'])
    if args.num_replicas_to_aggregate == -1:
        num_replicas_to_aggregate = num_workers
    else:
        num_replicas_to_aggregate = args.num_replicas_to_aggregate


def BasicConfig(args):
    if args.distributed:
        if args.task_id == 0:
            if not tf.gfile.Exists(args.train_dir):
                tf.gfile.MakeDirs(args.train_dir)
    else:
            if not tf.gfile.Exists(args.train_dir):
                tf.gfile.MakeDirs(args.train_dir)



def train_distibuted(args):
    res, cluster_spec = build_cluster_spec(args)
    if not res:
        ata_log("build_cluster_spec error")
        return

    res = DistributedConfig(args, cluster_spec)
    is_chief = (args.task_id == 0)

    # Ops are assigned to worker by default.
    with tf.device('/job:worker/task:%d' % args.task_id):
        # Variables and its related init/assign ops are assigned to ps.
        with slim.scopes.arg_scope(
                                   [slim.variables.variable, slim.variables.global_step],
                                   device=slim.variables.VariableDeviceChooser(num_parameter_servers)):

            #####################
            # data fetch config #
            #####################
            images, labels = image_processing.distorted_inputs(
                                dataset,
                                batch_size=args.batch_size,
                                num_preprocess_threads=args.num_preprocess_threads)

            #########################
            # LearningModule config #
            #########################
            LearningModuleConfig(args)

            ##################
            # Start Training #
            ##################
            StartTraining(args)


def train_single(args):



def train(args):
    if args.distributed:
        train_distibuted(args)
    else:
        train_single(args)


def parsers():
    parser = argparse.ArgumentParser()

    # ps_hosts
    parser.add_argument(
                        "--ps_hosts",
                        type=str,
                        default="",
                        help="Comma-separated list of hostname:port pairs"
    )

    args.worker_hosts
    args.job_name
    args.task_id
    args.dataset

    args.batch_size
    args.initial_learning_rate
    args.learning_rate_decay_factor
    args.num_epochs_per_decay
    args.rmsprop_decay
    args.rmsprop_momentum
    args.rmsprop_epsilon
    args.num_preprocess_threads


if __name__ == '__main__':
    train(args)
