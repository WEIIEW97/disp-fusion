"""Read-only native-RGB mono cache access and provenance checks."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def check_cached_rgb(meta, stem, digest, allow_legacy_unhashed=False):
    if meta.get('input_subdir') != 'RGB' or meta.get('undo_input_hflip') is not False:
        raise ValueError('native stored-RGB mono cache required')
    hashes = {r['stem']:r.get('input_sha256') for r in meta.get('inputs', [])}
    recorded = hashes.get(stem)
    if recorded is not None and recorded != digest:
        raise ValueError('input RGB hash differs from cache')
    if recorded is None and not allow_legacy_unhashed:
        raise ValueError('historical input hash absent; explicitly allow legacy cache to proceed')
    return dict(historical_input_hash_available=recorded is not None,
                historical_input_identity_verified=recorded == digest,
                limitation=None if recorded is not None else
                'Legacy cache has no historical input hash; current RGB hash cannot prove historical byte identity.')


class CachedMonoSource:
    """No model imports, GPU access, downloads or inference are performed."""

    def __init__(self, root, *, allow_legacy_unhashed=False):
        self.root = Path(root).resolve()
        self.metadata = json.loads((self.root/'run_info.json').read_text())
        self.allow_legacy_unhashed = allow_legacy_unhashed
        if self.metadata.get('input_subdir') != 'RGB' or self.metadata.get('undo_input_hflip') is not False:
            raise ValueError('only native stored-RGB caches are supported')

    @property
    def frames(self):
        return sorted(p.stem for p in self.root.glob('*.npy'))

    @property
    def name(self):
        return self.metadata.get('model', 'ZipDepth')

    def load(self, stem, rgb_path):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', stem):
            raise ValueError('frame must be a plain filename stem')
        rgb_hash = hashlib.sha256(Path(rgb_path).read_bytes()).hexdigest()
        provenance = check_cached_rgb(self.metadata, stem, rgb_hash, self.allow_legacy_unhashed)
        path = self.root/f'{stem}.npy'
        mono = np.load(path, allow_pickle=False).astype(np.float32)
        if mono.ndim != 2 or not np.isfinite(mono).all():
            raise ValueError('mono cache must contain finite 2-D arrays')
        provenance.update(rgb_sha256=rgb_hash, mono_prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          mono_root=str(self.root), checkpoint=self.metadata.get('checkpoint'))
        return mono, provenance
