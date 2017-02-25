
def distributed_config(cluster_spec):
  num_workers = len(cluster_spec.as_dict()['worker'])
  num_parameter_servers = len(cluster_spec.as_dict()['ps'])
  num_replicas_to_aggregate = num_workers
  is_chief = (FLAGS.task_id == 0)
    
