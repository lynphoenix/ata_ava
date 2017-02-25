import NetworkConfig
import OptimizerConfig
import LoggingConfig
from inception import inception_model as inception


def LearningModuleConfig(args, images, dataset):

    # Create a variable to count the number of train() calls. This equals the
    # number of updates applied to the variables.
    global_step = slim.variables.global_step()

    ##################
    #  compute loss  #
    ##################
    logits = inception.inference(images, dataset.num_classes(), for_training=True)

    # Add classification loss.
    inception.loss(logits, labels)
    # Gather all of the losses including regularization losses.
    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
    losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses, name='total_loss')


    if is_chief:
        # Compute the moving average of all individual losses and the
        # total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summmary to all individual losses and the total loss;
        # do the same for the averaged version of the losses.
        for l in losses + [total_loss]:
            loss_name = l.op.name
            # Name each loss as '(raw)' and name the moving average version of the
            # loss as the original loss name.
            tf.scalar_summary(loss_name + ' (raw)', l)
            tf.scalar_summary(loss_name, loss_averages.average(l))

        # Add dependency to compute loss_averages.
        with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)

    batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
    assert batchnorm_updates, 'Batchnorm updates are missing'
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    # Add dependency to compute batchnorm_updates.
    with tf.control_dependencies([batchnorm_updates_op]):
        total_loss = tf.identity(total_loss)

    #####################
    #  setup optimizer  #
    #####################
    # Track the moving averages of all trainable variables.
    # Note that we maintain a 'double-average' of the BatchNormalization
    # global statistics.
    # This is not needed when the number of replicas are small but important
    # for synchronous distributed training with tens of workers/replicas.
    exp_moving_averager = tf.train.ExponentialMovingAverage(
                            inception.MOVING_AVERAGE_DECAY, global_step)

    variables_to_average = (
                        tf.trainable_variables() + tf.moving_average_variables())



    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples_per_epoch() /
                             args.batch_size)

    # Decay steps need to be divided by the number of replicas to aggregate.
    decay_steps = int(num_batches_per_epoch * args.num_epochs_per_decay /
                      num_replicas_to_aggregate)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(args.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    args.learning_rate_decay_factor,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.RMSPropOptimizer(lr,
                                  args.rmsprop_decay,
                                  momentum=args.rmsprop_momentum,
                                  epsilon=args.rmsprop_epsilon)

    # Create synchronous replica optimizer.
    opt = tf.train.SyncReplicasOptimizer(
                                        opt,
                                        replicas_to_aggregate=num_replicas_to_aggregate,
                                        replica_id=FLAGS.task_id,
                                        total_num_replicas=num_workers,
                                        variable_averages=exp_moving_averager,
                                        variables_to_average=variables_to_average)

    # Compute gradients with respect to the loss.
    grads = opt.compute_gradients(total_loss)

    apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)

    ###################
    #  setup summary  #
    ###################
    # Add a summary to track the learning rate.
    tf.scalar_summary('learning_rate', lr)

    # Add histograms for model variables.
    for var in variables_to_average:
        tf.histogram_summary(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.histogram_summary(var.op.name + '/gradients', grad)


    with tf.control_dependencies([apply_gradients_op]):
        train_op = tf.identity(total_loss, name='train_op')

    return train_op

def StartTraining(args, is_chief, init_op, global_step, train_op):

    # Get chief queue_runners, init_tokens and clean_up_op, which is used to
    # synchronize replicas.
    # More details can be found in sync_replicas_optimizer.
    chief_queue_runners = [opt.get_chief_queue_runner()]
    init_tokens_op = opt.get_init_tokens_op()
    clean_up_op = opt.get_clean_up_op()

    # Create a saver.
    saver = tf.train.Saver()

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Build an initialization operation to run below.
    init_op = tf.initialize_all_variables()

    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    sv = tf.train.Supervisor(is_chief=is_chief,
                       logdir=args.train_dir,
                       init_op=init_op,
                       summary_op=None,
                       global_step=global_step,
                       saver=saver,
                       save_model_secs=args.save_interval_secs)

    tf.logging.info('%s Supervisor' % datetime.now())

    sess_config = tf.ConfigProto(
                                 allow_soft_placement=True,
                                 log_device_placement=args.log_device_placement)

    # Get a session.
    sess = sv.prepare_or_wait_for_session(target, config=sess_config)

    # Start the queue runners.
    queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
    sv.start_queue_runners(sess, queue_runners)
    tf.logging.info('Started %d queues for processing input data.',
                    len(queue_runners))

    if is_chief:
        sv.start_queue_runners(sess, chief_queue_runners)
        sess.run(init_tokens_op)

    # Train, checking for Nans. Concurrently run the summary operation at a
    # specified interval. Note that the summary_op and train_op never run
    # simultaneously in order to prevent running out of GPU memory.
    next_summary_time = time.time() + FLAGS.save_summaries_secs
    while not sv.should_stop():
        try:
            start_time = time.time()
            loss_value, step = sess.run([train_op, global_step])
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step > FLAGS.max_steps:
                break
            duration = time.time() - start_time

            if step % 30 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = ('Worker %d: %s: step %d, loss = %.2f'
                              '(%.1f examples/sec; %.3f  sec/batch)')
                tf.logging.info(format_str %
                                (FLAGS.task_id, datetime.now(), step, loss_value,
                                 examples_per_sec, duration))

            # Determine if the summary_op should be run on the chief worker.
            if is_chief and next_summary_time < time.time():
                tf.logging.info('Running Summary operation on the chief.')
                summary_str = sess.run(summary_op)
                sv.summary_computed(sess, summary_str)
                tf.logging.info('Finished running Summary operation.')

                # Determine the next time for running the summary.
                next_summary_time += FLAGS.save_summaries_secs
        except:
            if is_chief:
                tf.logging.info('About to execute sync_clean_up_op!')
                sess.run(clean_up_op)
            raise

    # Stop the supervisor.  This also waits for service threads to finish.
    sv.stop()

    # Save after the training ends.
    if is_chief:
    saver.save(sess,
               os.path.join(FLAGS.train_dir, 'model.ckpt'),
               global_step=global_step)
