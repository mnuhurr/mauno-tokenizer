from pathlib import Path

import argparse
import torch

from common import read_yaml, init_log
from dataset import MelTokenDataset, MelEmbeddingDataset
from models import Tokenizer, TokenizerConfig
from models import Model, ModelConfig
from models.utils import model_size
from trainer import Trainer
from utils import data_sizes


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default='settings.yaml', help='config file (default: settings.yaml)')
    parser.add_argument('--logdir', help='tensorboard logdir')

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = read_yaml(args.config)
    logger = init_log('pretrain')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    checkpoint_dir = Path(cfg.get('checkpoint_dir', cache_dir / 'checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    batch_size = cfg.get('batch_size', 16)
    num_workers = cfg.get('num_dataloader_workers', 8)

    learning_rate = cfg.get('learning_rate', 1e-4)
    weight_decay = cfg.get('weight_decay', 1e-2)
    clip_grad_norm = cfg.get('clip_grad_norm')
    grad_acc_steps = cfg.get('grad_acc_steps', 1)
    n_iters = cfg.get('n_iters', 3)
    n_model_epochs = cfg.get('n_model_epochs', 10)
    n_tokenizer_epochs = cfg.get('n_tokenizer_epochs', 10)
    log_interval = cfg.get('log_interval')
    patience = cfg.get('patience')
    n_pool_mdl_layers = cfg.get('n_pool_mdl_layers')

    # models
    n_enc_channels = cfg.get('n_enc_channels', 256)
    d_model = cfg.get('d_model', 128)
    n_layers = cfg.get('n_layers', 6)
    n_heads = cfg.get('n_heads', 8)
    dropout = cfg.get('dropout', 0.1)
    p_model_masking = cfg.get('p_model_masking', 0.0)
    #masking_length = cfg.get('masking_length', 50)
    p_tokenizer_masking = cfg.get('p_tokenizer_masking', 0.0)
    codebook_dim = cfg.get('codebook_dim', 64)
    codebook_size = cfg.get('codebook_size', 256)

    data_fn = cache_dir / 'kr-seq.hdf'

    n_seq, n_mels = data_sizes(data_fn)
    logger.info(f'{n_seq=}, {n_mels=}')

    ds_tok = MelTokenDataset(data_fn)
    ds_emb = MelEmbeddingDataset(data_fn)

    tok_loader = torch.utils.data.DataLoader(ds_tok, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    emb_loader = torch.utils.data.DataLoader(ds_emb, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    tok_cfg = TokenizerConfig(
        n_mels=n_mels,
        n_enc_channels=n_enc_channels,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        p_masking=p_tokenizer_masking,
        codebook_size=codebook_size,
        codebook_dim=codebook_dim)

    mdl_cfg = ModelConfig(
        n_mels=n_mels,
        n_enc_channels=n_enc_channels,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        #masking_length=masking_length,
        p_masking=p_model_masking,
        n_tokens=codebook_size)

    print(tok_cfg)
    print(mdl_cfg)

    tokenizer = Tokenizer(tok_cfg)
    model = Model(mdl_cfg)
    logger.info(f'model size {model_size(model) / 1e6:.1f}M')

    mdl_optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tok_optimizer = torch.optim.AdamW(tokenizer.parameters(), lr=learning_rate, weight_decay=weight_decay)

    pretrainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        token_loader=tok_loader,
        emb_loader=emb_loader,
        mdl_optimizer=mdl_optimizer,
        tok_optimizer=tok_optimizer,
        model_lr=learning_rate,
        tokenizer_lr=learning_rate,
        patience=patience,
        n_pool_mdl_layers=n_pool_mdl_layers,
        clip_grad_norm=clip_grad_norm,
        grad_acc_steps=grad_acc_steps,
        logger=logger,
        logdir=args.logdir,
        log_interval=log_interval,
        device=device)

    # main loop
    for it in range(n_iters):
        logger.info(f'begin iteration {it + 1}')
        # (1) update tokens and train model
        pretrainer.update_tokens()
        pretrainer.train_model(n_model_epochs)

        # (2) update embeddings and train tokenizer
        pretrainer.update_embeddings()
        pretrainer.train_tokenizer(n_tokenizer_epochs)

        # (3) save checkpoint
        mdl_ckpt = {
            'config': mdl_cfg,
            'state_dict': model.state_dict(),
        }
        tok_ckpt = {
            'config': tok_cfg,
            'state_dict': tokenizer.state_dict(),
        }

        torch.save(mdl_ckpt, checkpoint_dir / f'model-{it + 1:02d}.pt')
        torch.save(tok_ckpt, checkpoint_dir / f'tokenizer-{it + 1:02d}.pt')
        pretrainer.iteration += 1


if __name__ == '__main__':
    main()

