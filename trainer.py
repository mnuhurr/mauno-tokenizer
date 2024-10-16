from pathlib import Path
from tqdm import tqdm

import logging
import time
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import Tokenizer, Model


@torch.no_grad()
def _make_mask(starts: torch.Tensor, mask_seq_len: int, batch_len: int) -> torch.Tensor:
    mask = torch.ones(len(starts), batch_len, device=starts.device)

    for k, s in enumerate(starts):
        mask[k, s:s + mask_seq_len] = 0

    return mask


def masked_cross_entropy(y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(y_pred, y_true, reduction='none')
    return (mask.to(losses.dtype) * losses).sum() / mask.sum()


class Trainer:
    def __init__(self,
                 model: Model,
                 tokenizer: Tokenizer,
                 token_loader: torch.utils.data.DataLoader,
                 emb_loader: torch.utils.data.DataLoader,
                 mdl_optimizer: torch.optim.Optimizer,
                 tok_optimizer: torch.optim.Optimizer,
                 loss_tolerance: float = 0.0,
                 patience: int | None = None,
                 model_lr: float = 1.0e-3,
                 tokenizer_lr: float = 1.0e-3,
                 n_pool_mdl_layers: int | None = None,
                 clip_grad_norm: float | None = None,
                 grad_acc_steps: int = 1,
                 logger: logging.Logger | None = None,
                 log_interval: int | None = None,
                 logdir: str | Path | None = None,
                 device: torch.device | None = None):

        self.device = device if device is not None else torch.device('cpu')
        self.model = model.to(self.device)
        self.tokenizer = tokenizer.to(self.device)
        self.token_loader = token_loader
        self.emb_loader = emb_loader
        self.mdl_optimizer = mdl_optimizer
        self.tok_optimizer = tok_optimizer
        self.mdl_scheduler = None
        self.tok_scheduler = None
        self.model_lr = model_lr
        self.tokenizer_lr = tokenizer_lr

        self.n_pool_mdl_layers = n_pool_mdl_layers

        self.clip_grad_norm = clip_grad_norm
        self.grad_acc_steps = grad_acc_steps
        self.logger = logger
        self.log_interval = log_interval
        self.writer = SummaryWriter(log_dir=logdir) if logdir is not None else None

        self.iteration = 0
        self.model_step = 0
        self.tokenizer_step = 0

        self.max_patience = patience

    def _log(self, msg: str):
        if self.logger is not None:
            self.logger.info(msg)

    @torch.inference_mode()
    def update_tokens(self):
        self.tokenizer.eval()

        # check token sequence length
        x, _, _ = next(iter(self.token_loader))
        t, _ = self.tokenizer(x.to(self.device))
        seq_len = t.size(1)

        tokens = torch.zeros(len(self.token_loader.dataset), seq_len, dtype=torch.int64)

        for x, _, idx in tqdm(self.token_loader):
            x = x.to(self.device)

            tok, _ = self.tokenizer(x)
            tokens[idx] = tok.cpu()

        self.token_loader.dataset.tokens = tokens

    @torch.inference_mode()
    def update_embeddings(self):
        self.model.eval()

        # check dims
        x, _, _ = next(iter(self.emb_loader))
        _, z = self.model(x.to(self.device))
        _, seq_len, d_emb = z[0].shape

        embeddings = torch.empty(len(self.emb_loader.dataset), seq_len, d_emb)

        for x, _, idx in tqdm(self.token_loader):
            x = x.to(self.device)

            _, z, _ = self.model(x)

            if self.n_pool_mdl_layers is not None:
                z = torch.stack(z[-self.n_pool_mdl_layers:])
                z = F.normalize(z, p=2, dim=-1)
                z = z.mean(dim=0)
            else:
                #z = F.normalize(z[-1], p=2, dim=-1)
                #z = F.layer_norm(z[-1], (z[-1].size(-1),))
                z = z[-1]

            embeddings[idx] = z.cpu()

        self.emb_loader.dataset.embeddings = embeddings

    def train_model_epoch(self):
        self.model.train()
        train_loss = 0.0
        batch_t0 = time.perf_counter()
        scaler = torch.amp.GradScaler(device=self.device)

        self.mdl_optimizer.zero_grad()
        for batch, (x, y_true, _) in enumerate(self.token_loader):
            x = x.to(self.device)
            y_true = y_true.to(self.device)

            with torch.amp.autocast(device_type=self.device.type):
                y_pred, _, mask = self.model(x)
                #loss = F.cross_entropy(y_pred.permute(0, 2, 1), y_true)
                loss = masked_cross_entropy(y_pred.permute(0, 2, 1), y_true, mask)

            train_loss += loss.item()
            scaler.scale(loss / self.grad_acc_steps).backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.clip_grad_norm is not None:
                    scaler.unscale_(self.mdl_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)

                scaler.step(self.mdl_optimizer)
                scaler.update()
                self.mdl_optimizer.zero_grad()

            if self.mdl_scheduler is not None:
                self.mdl_scheduler.step()

            if self.log_interval is not None and batch % self.log_interval == 0:
                t_batch = int(1000 * (time.perf_counter() - batch_t0) / self.log_interval)
                current_lr = self.mdl_optimizer.param_groups[0]['lr']
                print(f'batch {batch:4d}/{len(self.token_loader)} - {t_batch} ms/batch - current lr {current_lr:.4g} - training loss {loss.item():.4f}')

                if self.writer is not None:
                    self.writer.add_scalar('model/loss', loss.item(), self.model_step)
                    self.writer.add_scalar('model/lr', current_lr, self.model_step)

                batch_t0 = time.perf_counter()

            self.model_step += 1

        return train_loss / len(self.token_loader)

    def train_model(self, n_epochs: int):
        """ run the model training for an iteration """
        # reset scheduler
        patience = self.max_patience
        best_loss = float('inf')

        self.mdl_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.mdl_optimizer, max_lr=self.model_lr, total_steps=n_epochs * len(self.token_loader), pct_start=0.1)

        for epoch in range(n_epochs):
            train_loss = self.train_model_epoch()
            self._log(f'model epoch {epoch + 1} - training loss {train_loss:.4f}')

            if self.writer is not None:
                self.writer.add_scalar(f'iter_{self.iteration}/model_loss', train_loss, epoch)

            if train_loss < best_loss:
                best_loss = train_loss
                patience = self.max_patience

            elif patience is not None:
                patience -= 1
                if patience <= 0:
                    self._log('results not improving, stopping...')
                    break

    def train_tokenizer_epoch(self):
        """ train the tokenizer for a single epoch """
        self.tokenizer.train()
        train_loss = 0.0
        batch_t0 = time.perf_counter()
        scaler = torch.amp.GradScaler(device=self.device)

        self.tok_optimizer.zero_grad()
        for batch, (x, z_true, _) in enumerate(self.emb_loader):
            x = x.to(self.device)
            z_true = z_true.to(self.device)

            with torch.amp.autocast(device_type=self.device.type):
                _, z_pred = self.tokenizer(x)
                commit_loss = torch.mean(self.tokenizer.commit_loss) if self.tokenizer.commit_loss is not None else torch.tensor(0.0, device=self.device)
                mse_loss = F.mse_loss(z_pred, z_true)
                loss = mse_loss + commit_loss

            train_loss += loss.item()
            scaler.scale(loss / self.grad_acc_steps).backward()

            if (batch + 1) % self.grad_acc_steps == 0:
                if self.clip_grad_norm is not None:
                    scaler.unscale_(self.tok_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.tokenizer.parameters(), self.clip_grad_norm)

                scaler.step(self.tok_optimizer)
                scaler.update()
                self.tok_optimizer.zero_grad()

            if self.tok_scheduler is not None:
                self.tok_scheduler.step()

            if self.log_interval is not None and batch % self.log_interval == 0:
                t_batch = int(1000 * (time.perf_counter() - batch_t0) / self.log_interval)
                current_lr = self.tok_optimizer.param_groups[0]['lr']
                print(f'batch {batch:4d}/{len(self.emb_loader)} - {t_batch} ms/batch - current lr {current_lr:.4g} - mse loss {mse_loss.item():.4f} - commit loss {commit_loss.item():.4f}')

                if self.writer is not None:
                    self.writer.add_scalar('tokenizer/loss', loss.item(), self.tokenizer_step)
                    self.writer.add_scalar('tokenizer/lr', current_lr, self.tokenizer_step)

                batch_t0 = time.perf_counter()

            self.tokenizer_step += 1

        return train_loss / len(self.token_loader)

    def train_tokenizer(self, n_epochs: int):
        # reset scheduler
        self.tok_scheduler = torch.optim.lr_scheduler.OneCycleLR(self.tok_optimizer, max_lr=self.tokenizer_lr, total_steps=n_epochs * len(self.emb_loader), pct_start=0.1)

        best_loss = float('inf')
        patience = self.max_patience
        for epoch in range(n_epochs):
            train_loss = self.train_tokenizer_epoch()
            self._log(f'tokenizer epoch {epoch + 1} - training loss {train_loss:.4f}')

            if self.writer is not None:
                self.writer.add_scalar(f'iter_{self.iteration}/tokenizer_loss', train_loss, epoch)

            if train_loss < best_loss:
                best_loss = train_loss
                patience = self.max_patience

            elif patience is not None:
                patience -= 1
                if patience <= 0:
                    self._log('results not improving, stopping...')
                    break

