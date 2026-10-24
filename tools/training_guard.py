"""Fixed batch/epoch protocol for pinned Ultralytics 8.4.183; no dataset access."""
class TrainingProtocolError(ValueError):
    pass


def fixed_batch_trainer(base, expected_batch, validation_batch=1):
    if expected_batch < 1 or validation_batch < 1:
        raise TrainingProtocolError('Batch sizes must be positive integers')

    class FixedBatchTrainer(base):
        def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
            if mode == 'val':
                batch_size = validation_batch
            elif int(batch_size) != expected_batch:
                raise TrainingProtocolError(f'Refusing changed training batch={batch_size}; expected {expected_batch}. No reduced-batch training is allowed.')
            return super().get_dataloader(dataset_path,batch_size=batch_size,rank=rank,mode=mode)
    return FixedBatchTrainer


def install_batch_guard(wrapper, expected_batch):
    def check(trainer):
        if int(trainer.batch_size) != expected_batch or int(trainer.args.batch) != expected_batch:
            raise TrainingProtocolError(f'Actual batch changed: {trainer.batch_size}; expected {expected_batch}')

    def before_batch(trainer):
        check(trainer)
        # Pinned trainer catches CUDA RuntimeError and retries if this is <3.
        # Arm the existing retry limit for this batch; an OOM propagates immediately.
        if not hasattr(trainer,'_oom_retries'):
            raise TrainingProtocolError('Unsupported Ultralytics OOM policy: review pinned version before training')
        trainer._oom_retries = 3

    def after_batch(trainer):
        check(trainer)
        trainer._oom_retries = 0  # Successful batch: do not trigger an epoch retry.

    wrapper.add_callback('on_train_start',check)
    wrapper.add_callback('on_train_batch_start',before_batch)
    wrapper.add_callback('on_train_batch_end',after_batch)
    return check


def completed_training(trainer, config):
    actual = dict(batch=int(trainer.batch_size), effective_batch=int(trainer.args.nbs),
                  validation_batch=int(trainer.test_loader.batch_size),
                  epochs_completed=int(trainer.epoch)+1,epochs_requested=int(trainer.args.epochs),
                  seed=int(trainer.args.seed),patience=int(trainer.args.patience))
    expected = dict(batch=int(config['batch']),effective_batch=int(config['effective_batch']),
                    validation_batch=int(config.get('validation_batch',1)),epochs_completed=int(config['epochs']),
                    epochs_requested=int(config['epochs']),seed=int(config['seed']),patience=0)
    if actual != expected:
        raise TrainingProtocolError(f'Incomplete or changed training protocol: actual={actual}, expected={expected}')
    return actual
