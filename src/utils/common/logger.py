import os

from .dist import is_main_process

class TextLogger():
    def __init__(self, save, filename):
        if is_main_process():
            self.f = os.path.join(save, filename)

    def write(self, log, print_log=True):
        if is_main_process():
            if print_log:
                print(log)
            with open(self.f, 'a') as f:
                f.write(log+'\n')
            return f.close()


class TensorboardLogger():
    def __init__(self, log_dir):
        if is_main_process():
            from torch.utils.tensorboard import SummaryWriter
            self.tb_logger = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, name, value, current_iter):
        if is_main_process():
            self.tb_logger.add_scalar(name, value, current_iter)
        return


class WandbLogger():
    def __init__(self, project, name, config=None):
        if is_main_process():
            import wandb
            self.run = wandb.init(project=project, name=name, config=config, resume='allow')
        self._is_main = is_main_process()

    def add_scalar(self, name, value, current_iter):
        if self._is_main:
            import wandb
            wandb.log({name: value}, step=current_iter)

    def finish(self):
        if self._is_main:
            import wandb
            wandb.finish()
