from pathlib import Path
from tqdm import tqdm

import h5py
import librosa
import numpy as np
import torch

from common import read_yaml, init_log


def time_seconds(s: str) -> int:
    m, s = s.split(':')
    return 60 * int(m) + int(s)


def read_meta(filename: str | Path) -> dict[str, tuple[int, int]]:
    events = {}
    lines = Path(filename).read_text().splitlines()

    for line in lines:
        parts = line.split()

        fn = f'kansanradio-{parts[0]}.mp3'

        start = time_seconds(parts[1])
        end = time_seconds(parts[3])
        events[fn] = (start, end)

    return events


def povey_window(window_length: int) -> np.ndarray:
    w = 2 * np.pi * np.arange(window_length) / (window_length - 1)
    return (0.5 - 0.5 * np.cos(w))**0.85


def main(config_fn='settings.yaml'):
    cfg = read_yaml(config_fn)
    logger = init_log('preapre-data')

    cache_dir = Path(cfg.get('cache_dir', 'cache'))
    cache_dir.mkdir(exist_ok=True, parents=True)

    kr_dir = Path(cfg.get('kr_dir'))

    # audio features
    sample_rate = cfg.get('sample_rate', 32000)
    n_fft = cfg.get('n_fft', 1024)
    hop_length = cfg.get('hop_length', 640)
    n_mels = cfg.get('n_mels', 64)
    f_min = cfg.get('f_min', 0.0)
    f_max = cfg.get('f_max', sample_rate / 2)
    win_length = cfg.get('win_length')
    window = 'hann' if win_length is None else povey_window(win_length)
    logger.info(f'{sample_rate=}, {n_fft=}, {hop_length=}, {n_mels=}, {f_min=}, {f_max=}, {win_length=}')

    sequence_length = cfg.get('sequence_length', 1000)
    sequence_hop = cfg.get('sequence_hop', 500)
    logger.info(f'{sequence_length=}, {sequence_hop=}')

    # read mauno occurrences
    events = read_meta(kr_dir / 'kr-mauno.txt')

    filenames = sorted((kr_dir / 'kr').glob('*.mp3'))

    sz = 1000

    train_idx = []

    hop_len_secs = hop_length / sample_rate

    with h5py.File(cache_dir / 'kr-seq.hdf', 'w') as hdf_file:
        mels = hdf_file.create_dataset(
            'mels',
            (sz, n_mels, sequence_length),
            maxshape=(10_000_000, n_mels, sequence_length),
            chunks=(1, n_mels, sequence_length))

        event = hdf_file.create_dataset(
            'event',
            (sz, sequence_length),
            maxshape=(10_000_000, sequence_length),
            chunks=(1, sequence_length))

        idx = 0

        for fn in tqdm(filenames):
            y, _ = librosa.load(fn, sr=sample_rate, mono=True)

            m = librosa.feature.melspectrogram(
                y=y,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                n_mels=n_mels,
                fmin=f_min,
                fmax=f_max)

            e = np.zeros((m.shape[1],))
            has_event = False

            if fn.name in events:
                t0, t1 = events[fn.name]

                start = int(t0 / hop_len_secs)
                end = int(t1 / hop_len_secs)

                e[start:end + 1] = 1
                has_event = True

            n_seq = np.ceil((m.shape[1] - sequence_length) / sequence_hop).astype(np.int32) + 1

            for k in range(n_seq):
                start = k * sequence_hop
                m_seq = m[:, start:start + sequence_length]
                e_seq = e[start:start + sequence_length]

                assert m_seq.shape[1] == e_seq.shape[0]
                seg_len = m_seq.shape[1]

                mels[idx, :, :seg_len] = m_seq
                event[idx, :seg_len] = e_seq

                if has_event:
                    train_idx.append(idx)

                idx += 1

                if idx == sz:
                    sz += 1000
                    mels.resize((sz, n_mels, sequence_length))
                    event.resize((sz, sequence_length))

        mels.resize((idx, n_mels, sequence_length))
        event.resize((idx, sequence_length))

    torch.save(train_idx, cache_dir / 'train_idx.pt')
    logger.info(f'serialized {idx} segments')


if __name__ == '__main__':
    main()

