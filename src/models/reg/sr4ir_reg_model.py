import os
import os.path as osp
import torch
import warnings
warnings.filterwarnings('ignore')
# to ignore below warning message from COCO evaluator:
# UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or
# sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).

from archs import build_network
from losses import build_loss
from math import ceil
from torch.nn.functional import pad
from utils.common import save_on_master, reduce_across_processes

from .base_model import BaseModel


def make_model(opt):
    return SR4IRRegressionModel(opt)


class SR4IRRegressionModel(BaseModel):
    """Semantic Segmentation model using Super-Resolution for Image Recognition."""

    def __init__(self, opt):
        super().__init__(opt)
        
        # define network sr
        self.sr_is_trainable = self.is_train and opt['train'].get('sr_is_trainable', True)
        opt['network_sr']['scale'] = self.scale
        self.net_sr = build_network(opt['network_sr'], self.text_logger, tag='net_sr')
        self.load_network(self.net_sr, name='network_sr', tag='net_sr')
        self.net_sr = self.model_to_device(self.net_sr, is_trainable=self.sr_is_trainable)
        self.print_network(self.net_sr, tag='net_sr')
        
        # define network regression for hr (not trainable)
        self.reg_is_trainable = self.is_train and opt['train'].get('reg_is_trainable', True)
        self.net_reg = build_network(opt['network_reg'], self.text_logger, task=self.task, tag='net_reg')
        self.load_network(self.net_reg, name='network_reg', tag='net_reg')
        self.net_reg = self.model_to_device(self.net_reg, is_trainable=self.reg_is_trainable)
        self.print_network(self.net_reg, tag='net_reg')
        
    def set_mode(self, mode):
        if mode == 'train':
            self.net_sr.train()
            self.net_reg.train()
        elif mode == 'eval':
            self.net_sr.eval()
            self.net_reg.eval()    
        else:
            raise NotImplementedError(f"mode {mode} is not supported")
        
    def init_training_settings(self, data_loader_train):
        self.set_mode(mode='train')
        train_opt = self.opt['train']
        
        # phase 1
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt'], self.text_logger).to(self.device)
        
        if train_opt.get('tdp_opt'):
            # task driven perceptual loss
            self.cri_tdp = build_loss(train_opt['tdp_opt'], self.text_logger).to(self.device)
            
        # phase 2
        if train_opt.get('huber_sr_opt'):
            self.cri_huber_sr = build_loss(train_opt['huber_sr_opt'], self.text_logger).to(self.device)
        
        if train_opt.get('huber_hr_opt'):
            self.cri_huber_hr = build_loss(train_opt['huber_hr_opt'], self.text_logger).to(self.device)
            
        if train_opt.get('huber_cqmix_opt'):
            self.cri_huber_cqmix = build_loss(train_opt['huber_cqmix_opt'], self.text_logger).to(self.device)
        
        # set up optimizers and schedulers
        self.setup_optimizers()
        # use max_train_iters (not true dataloader length) so scheduler periods
        # align with actual steps per epoch, enabling epoch-level restart control
        max_train_iters = train_opt.get('max_iters', 150)
        self.setup_schedulers(max_train_iters, name='sr', optimizer=self.optimizer_sr)
        self.setup_schedulers(max_train_iters, name='reg', optimizer=self.optimizer_reg)
        
        # mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda')

        # set up saving directories
        os.makedirs(osp.join(self.exp_dir, 'models'), exist_ok=True)
        os.makedirs(osp.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        
        # eval freq
        self.eval_freq = train_opt.get('eval_freq', 1)
        
        # warmup epoch
        self.warmup_epoch = train_opt.get('warmup_epoch', -1)
        self.text_logger.write("NOTICE: total epoch: {}, warmup epoch: {}".format(train_opt['epoch'], self.warmup_epoch))
        
    def setup_optimizers(self):
        train_opt = self.opt['train']
        
        # optimizer sr
        optim_type = train_opt['optim_sr'].pop('type')
        self.optimizer_sr = self.get_optimizer(optim_type, self.net_sr.parameters(), **train_opt['optim_sr'])
        self.optimizers.append(self.optimizer_sr)
        
        # optimizer reg
        self.text_logger.write('NOTICE: net_reg is trainable')
        optim_type = train_opt['optim_reg'].pop('type')
        # 冻结 DAv2 backbone，只训练 neck + head
        bare_reg = self.get_bare_model(self.net_reg)
        for p in bare_reg.backbone.parameters():
            p.requires_grad = False
        self.optimizer_reg = self.get_optimizer(optim_type, filter(lambda p: p.requires_grad, bare_reg.parameters()), **train_opt['optim_reg'])
        self.optimizers.append(self.optimizer_reg)
        
    def train_one_epoch(self, data_loader_train, train_sampler, epoch):
        from tqdm import tqdm
        self.set_mode(mode='train')

        if self.dist:
            train_sampler.set_epoch(epoch)
        if epoch < self.warmup_epoch + 1:
            self.text_logger.write("NOTICE: Doing warm-up")

        torch.cuda.empty_cache()

        max_train_iters = self.opt['train'].get('max_iters', 150)
        loss_pix_sum = 0.0
        loss_tdp_sum = 0.0
        loss_huber_hr_sum = 0.0
        n_iters = 0

        pbar = tqdm(data_loader_train, desc=f"Epoch {epoch}", leave=True,
                    dynamic_ncols=True, total=min(len(data_loader_train), max_train_iters))
        for batch in pbar:
            if n_iters >= max_train_iters:
                break
            img_ls, img_ps, target = batch
            img_ls = img_ls.squeeze(0).to(self.device)
            img_ps = img_ps.squeeze(0).to(self.device)
            target = target.squeeze(0).to(self.device)

            # phase 1: update net_sr, freeze net_reg
            for p in self.net_reg.parameters(): p.requires_grad = False
            self.optimizer_sr.zero_grad()
            img_sr = self.net_sr(img_ls.float())  # SwinIR needs float32; AMP causes NaN in attention
            l_total_sr = None
            if hasattr(self, 'cri_pix'):
                l_pix = self.cri_pix(img_sr, img_ps.float())
                loss_pix_sum += l_pix.item()
                l_total_sr = l_pix
            with torch.amp.autocast('cuda'):
                if epoch > self.warmup_epoch:
                    if hasattr(self, 'cri_tdp'):
                        self.net_reg.eval()



                        # net_reg (HuggingFace wrapper) does not support return_feats;
                        # use a forward hook to capture neck.fusion_stage output instead
                        feats = {}
                        bare_reg = self.get_bare_model(self.net_reg)
                        hook = bare_reg.neck.fusion_stage.register_forward_hook(
                            lambda _m, _inp, out: feats.update({'out': out})
                        )
                        self.net_reg(img_sr)
                        feat_sr = {'C5': feats['out']}
                        self.net_reg(img_ps)
                        feat_hr = {'C5': feats['out']}
                        hook.remove()



                        self.net_reg.train()
                        l_tdp = self.cri_tdp(feat_sr, feat_hr)
                        loss_tdp_sum += l_tdp.item()
                        l_total_sr = l_tdp if l_total_sr is None else l_total_sr + l_tdp
            if l_total_sr is not None:
                self.scaler.scale(l_total_sr).backward()
                self.scaler.step(self.optimizer_sr)
            for p in self.net_reg.parameters(): p.requires_grad = True
            # restore backbone freeze
            bare_reg = self.get_bare_model(self.net_reg)
            for p in bare_reg.backbone.parameters(): p.requires_grad = False


            # phase 2: update net_reg neck+head only (backbone stays frozen)
            self.optimizer_reg.zero_grad()
            target_reg = target.squeeze(1)  # (B,1,H,W) -> (B,H,W)
            with torch.no_grad():
                img_sr_detached = self.net_sr(img_ls.float())
            with torch.amp.autocast('cuda'):
                l_total_reg = 0
                if hasattr(self, 'cri_huber_hr'):
                    pred_sr = self.net_reg(img_sr_detached).predicted_depth
                    l_huber_sr = self.cri_huber_hr(pred_sr, target_reg)
                    l_total_reg += l_huber_sr
                    loss_huber_hr_sum += l_huber_sr.item()
            self.scaler.scale(l_total_reg).backward()
            self.scaler.step(self.optimizer_reg)
            self.scaler.update()  # called once per iteration, after all optimizer steps

            self.update_learning_rate()
            n_iters += 1

        lr_sr = self.optimizer_sr.param_groups[0]["lr"]
        lr_reg = self.optimizer_reg.param_groups[0]["lr"]
        avg_pix = loss_pix_sum / max(n_iters, 1)
        avg_tdp = loss_tdp_sum / max(n_iters, 1)
        avg_loss = loss_huber_hr_sum / max(n_iters, 1)
        self.text_logger.write(
            f"Epoch {epoch}  l_pix {avg_pix:.4f}  l_tdp {avg_tdp:.4f}  "
            f"loss_huber_sr {avg_loss:.4f}  lr_sr {lr_sr:.2e}  lr_reg {lr_reg:.2e}"
        )
        from utils.common.dist import is_main_process
        if is_main_process():
            import wandb
            wandb.log({
                'train/l_pix': avg_pix,
                'train/l_tdp': avg_tdp,
                'train/loss_huber_sr': avg_loss,
                'train/lr_sr': lr_sr,
                'train/lr_reg': lr_reg,
            }, step=epoch, commit=False)
        return
            
    @torch.inference_mode()
    def evaluate(self, data_loader_test, epoch=0, **kwargs):
        from tqdm import tqdm
        import random as _random

        if hasattr(self, 'eval_freq') and (epoch % self.eval_freq != 0):
            return

        self.set_mode(mode='eval')

        mae_sum = torch.tensor(0.0, device=self.device)
        rmse_sum = torch.tensor(0.0, device=self.device)
        target_sum = torch.tensor(0.0, device=self.device)
        target_sq_sum = torch.tensor(0.0, device=self.device)
        loss_sum = 0.0
        num_valid = torch.tensor(0, device=self.device)
        num_processed_samples = 0

        max_eval_iters = self.opt['train'].get('max_eval_iters', 60)
        viz_pool = []
        viz_total = 6
        n_eval_iters = 0

        pbar = tqdm(data_loader_test, desc=f"Eval  {epoch}", leave=True,
                    dynamic_ncols=True, total=min(len(data_loader_test), max_eval_iters))
        for batch in pbar:
            if n_eval_iters >= max_eval_iters:
                break
            img_ls, img_ps, target = batch
            img_ls = img_ls.squeeze(0).to(self.device)
            img_ps = img_ps.squeeze(0).to(self.device)
            target = target.squeeze(0).to(self.device)

            img_sr = self.net_sr(img_ls.float())
            pred_sr = self.net_reg(img_sr).predicted_depth

            target = target.squeeze(1)  # (B, 1, H, W) -> (B, H, W)
            valid_mask = torch.isfinite(target)
            diff = (pred_sr - target)[valid_mask]
            t_valid = target[valid_mask]
            mae_sum += diff.abs().sum()
            rmse_sum += (diff ** 2).sum()
            target_sum += t_valid.sum()
            target_sq_sum += (t_valid ** 2).sum()
            num_valid += valid_mask.sum()

            if hasattr(self, 'cri_huber_hr'):
                loss_sum += self.cri_huber_hr(pred_sr, target).item()

            num_processed_samples += img_ls.shape[0]

            b = img_ps.shape[0]
            for i in range(b):
                sample = (
                    img_sr[i].detach().cpu(),
                    img_ps[i].detach().cpu(),
                    pred_sr[i].detach().cpu(),
                    target[i].detach().cpu(),
                )
                if len(viz_pool) < viz_total:
                    viz_pool.append(sample)
                else:
                    j = _random.randint(0, num_processed_samples + i)
                    if j < viz_total:
                        viz_pool[j] = sample

            n_eval_iters += 1

        num_valid = reduce_across_processes(num_valid)
        mae_sum = reduce_across_processes(mae_sum)
        rmse_sum = reduce_across_processes(rmse_sum)
        target_sum = reduce_across_processes(target_sum)
        target_sq_sum = reduce_across_processes(target_sq_sum)

        mae = (mae_sum / num_valid).item()
        rmse = ((rmse_sum / num_valid) ** 0.5).item()
        target_mean = target_sum / num_valid
        ss_tot = target_sq_sum - num_valid * target_mean ** 2
        r2 = (1.0 - rmse_sum / ss_tot).item() if ss_tot > 0 else float('nan')
        avg_loss = loss_sum / max(n_eval_iters, 1)

        self.text_logger.write(f"Eval  epoch {epoch}  loss {avg_loss:.4f}  r2 {r2:.4f}")

        if self.is_train:
            self._log_eval_to_wandb(avg_loss, mae, rmse, r2, viz_pool, epoch)
        return

    def _log_eval_to_wandb(self, avg_loss, mae, rmse, r2, viz_pool, epoch):
        from utils.common.dist import is_main_process
        if not is_main_process():
            return
        import wandb
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        log_dict = {
            'val/loss': avg_loss,
            'val/mae': mae,
            'val/rmse': rmse,
            'val/r2': r2,
        }

        def to_rgb_np(t):
            arr = t.float().permute(1, 2, 0).numpy()
            arr = arr - arr.min()
            rng = arr.max()
            if rng > 1e-8:
                arr = arr / rng
            return np.clip(arr, 0, 1)

        num_samples = len(viz_pool)
        ncols = 4
        fig, axes = plt.subplots(num_samples, ncols, figsize=(ncols * 4, 5 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i, (img_sr, img_ps, pred_sr, tgt) in enumerate(viz_pool):
            rgb_sr = to_rgb_np(img_sr)
            rgb_ps = to_rgb_np(img_ps)

            pred_np = pred_sr.float().numpy()
            tgt_np  = tgt.float().numpy()
            pred_mean, pred_std = np.mean(pred_np), np.std(pred_np)
            pred_min,  pred_max = np.min(pred_np),  np.max(pred_np)
            tgt_mean,  tgt_std  = np.mean(tgt_np),  np.std(tgt_np)
            tgt_min,   tgt_max  = np.min(tgt_np),   np.max(tgt_np)
            vmin = min(pred_np.min(), tgt_np.min())
            vmax = max(pred_np.max(), tgt_np.max())

            axes[i, 0].imshow(rgb_sr)
            axes[i, 0].set_title(f'Sample {i+1}: img_sr RGB')

            axes[i, 1].imshow(rgb_ps)
            axes[i, 1].set_title(f'Sample {i+1}: img_ps RGB')

            axes[i, 2].imshow(pred_np, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 2].set_title(f'Pred ({pred_mean:.1f}±{pred_std:.1f}); {pred_min:.1f}-{pred_max:.1f}')

            axes[i, 3].imshow(tgt_np, cmap='viridis', vmin=vmin, vmax=vmax)
            axes[i, 3].set_title(f'Target ({tgt_mean:.1f}±{tgt_std:.1f}); {tgt_min:.1f}-{tgt_max:.1f}')

        plt.tight_layout()
        log_dict["val/sample_grid"] = wandb.Image(fig)
        plt.close(fig)

        # scatter plot
        all_preds   = np.concatenate([s[2].float().numpy().flatten() for s in viz_pool])
        all_targets = np.concatenate([s[3].float().numpy().flatten() for s in viz_pool])
        fig_scatter, ax = plt.subplots()
        ax.scatter(all_targets, all_preds, alpha=0.5, s=2)
        ax.set_xlabel('Target')
        ax.set_ylabel('Pred')
        maxi = max(all_targets.max(), all_preds.max())
        ax.set_xlim(0, maxi)
        ax.set_ylim(0, maxi)
        ax.set_title('Scatter: Pred vs Target (all samples)')
        log_dict["val/scatter"] = wandb.Image(fig_scatter)
        plt.close(fig_scatter)

        wandb.log(log_dict, step=epoch, commit=True)


    def save(self, epoch):            
        checkpoint = {"epoch": epoch,
                      "opt": self.opt,
                      "net_sr": self.get_bare_model(self.net_sr).state_dict(),
                      "net_reg": self.get_bare_model(self.net_reg).state_dict(),
                      'schedulers': [],
                      }
        for s in self.schedulers:
            checkpoint['schedulers'].append(s.state_dict())
                
        if epoch % self.opt['train']['save_freq'] == 0:
            save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_{:03d}.pth".format(epoch)))
            save_on_master(self.get_bare_model(self.net_reg).state_dict(), osp.join(self.exp_dir, 'models', "net_reg_{:03d}.pth".format(epoch)))
            save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_{:03d}.pth".format(epoch)))
            
        save_on_master(self.get_bare_model(self.net_sr).state_dict(), osp.join(self.exp_dir, 'models', "net_sr_latest.pth"))
        save_on_master(self.get_bare_model(self.net_reg).state_dict(), osp.join(self.exp_dir, 'models', "net_reg_latest.pth"))
        save_on_master(checkpoint, osp.join(self.exp_dir, 'checkpoints', "checkpoint_latest.pth"))
        return

    def _debug_visualize_batch(self, img_ls, img_ps, target):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vis_dir = osp.join(self.exp_dir, 'debug_data')
        os.makedirs(vis_dir, exist_ok=True)

        def log_stats(name, t):
            valid = t[torch.isfinite(t)]
            self.text_logger.write(
                f"[DEBUG] {name}: shape={list(t.shape)}, "
                f"min={valid.min().item():.4f}, max={valid.max().item():.4f}, "
                f"mean={valid.mean().item():.4f}, std={valid.std().item():.4f}, "
                f"nan={torch.isnan(t).sum().item()}, inf={torch.isinf(t).sum().item()}"
            )

        log_stats('img_ls', img_ls)
        log_stats('img_ps', img_ps)
        log_stats('target', target)

        def save_grid(tensor, path, cmap='viridis'):
            # tensor: [B, C, H, W] or [B, H, W]
            t = tensor.cpu().float()
            if t.dim() == 4:
                t = t[0]  # first sample, [C,H,W]
                if t.shape[0] >= 3:
                    img = t[:3].permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                    plt.imsave(path, img)
                else:
                    plt.imsave(path, t[0].numpy(), cmap=cmap)
            else:
                t = t[0].numpy()  # first sample [H,W]
                plt.imsave(path, t, cmap=cmap)

        save_grid(img_ls, osp.join(vis_dir, 'img_ls.png'))
        save_grid(img_ps, osp.join(vis_dir, 'img_ps.png'))
        save_grid(target, osp.join(vis_dir, 'target.png'), cmap='plasma')
        self.text_logger.write(f"[DEBUG] Visualizations saved to {vis_dir}")
