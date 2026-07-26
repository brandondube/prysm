"""Sample data for prysm tests and documentation."""
import shutil
import os
from pathlib import Path
from urllib.request import urlopen

baseremote = r'https://github.com/brandondube/prysm/raw/v0.21.1/sample_files/'
baselocal = Path(__file__).resolve()
bundled_root = baselocal.parent.parent / 'prysm-sampledata'
cache_root = Path(os.environ.get(
    'PRYSM_SAMPLE_DATA_DIR',
    Path.home() / '.cache' / 'prysm' / 'sample-data',
)).expanduser()
root = bundled_root if bundled_root.is_dir() else cache_root


def fetch_if_not_present(local, remote):
    """Fetch a file from github if it is not present in the local filesystem."""
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(remote) as response, open(local, 'wb') as fid:
            shutil.copyfileobj(response, fid)

    return local


class SampleFiles:
    """Sample files for prysm."""
    dat = 'valid_zygo_dat_file.dat'

    def __call__(self, dtype_or_filename):
        """Get the path of a sample file."""
        dtype_or_filename = str(dtype_or_filename).lower()
        if hasattr(self, dtype_or_filename):
            filename = getattr(self, dtype_or_filename)
            local = (root / getattr(self, dtype_or_filename)).absolute()
            remote = baseremote + filename
            return fetch_if_not_present(local, remote)

        else:
            local = root / dtype_or_filename
            remote = baseremote + dtype_or_filename
            return fetch_if_not_present(local, remote)


sample_files = SampleFiles()
