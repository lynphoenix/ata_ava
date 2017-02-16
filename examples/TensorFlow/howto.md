# Usages for TensorFlow Commands

## 分布式训练 ImageNet

### 启动训练步骤
* 启动N个TensorFlow镜像，并clone github.com/tensorflow/models (ataraxia/tftest:v0.1，其TensorFlow版本是0.12.1)

* 添加inception的路径

  export PYTHONPATH=/workspace/models/inception/:$PYTHONPATH


* 创建 /workspace/models/inception/__init__.py 和 /workspace/models/inception/slim/__init__.py 文件，如果没有这两个文件，inception和slim两个模块无法import

* 分别在和ps和worker服务器上运行分布式训练的ps和worker脚本，开始训练

  python inception/imagenet_distributed_train.py --batch_size=64 --data_dir=/disk2/data/ILSVRC2012/records/ --job_name="worker" --task_id=0 --ps_hosts='10.130.170.184:2222' --worker_hosts='10.130.224.136:2222,10.130.224.149:2222' --GLOG_logtostderr=1 --train_dir=/disk2/imagenet-train/

  python inception/imagenet_distributed_train.py --batch_size=64 --data_dir=/disk2/data/ILSVRC2012/records/ --job_name="worker" --task_id=1 --ps_hosts='10.130.170.184:2222' --worker_hosts='10.130.224.136:2222,10.130.224.149:2222' --GLOG_logtostderr=1 --train_dir=/disk2/imagenet-train/

  python inception/imagenet_distributed_train.py --batch_size=64 --data_dir=/disk2/data/ILSVRC2012/records/ --job_name="ps" --task_id=0 --ps_hosts='10.130.170.184:2222' --worker_hosts='10.130.224.136:2222,10.130.224.149:2222' --GLOG_logtostderr=1 --train_dir=/disk2/imagenet-train/


### Tips
* kirk默认的8G内存会导致out-of-memory错误，会直接将训练kill掉，而且查看不到任何相关信息。32GB内存可用。
