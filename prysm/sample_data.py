"""Sample data for prysm tests and documentation."""
import shutil
from pathlib import Path
from urllib.request import urlopen

baseremote = r'https://github.com/brandondube/prysm/raw/master/sample_files/'
baselocal = Path(__file__)
root = baselocal.parent / '..' / 'prysm-sampledata'
root.mkdir(exist_ok=True)


def fetch_if_not_present(local, remote):
    """Fetch a file from github if it is not present in the local filesystem."""
    if not local.exists():
        with urlopen(remote) as response, open(local, 'wb') as fid:
            shutil.copyfileobj(response, fid)

    return local


class SampleFiles:
    """Sample files for prysm."""
    dat = 'valid_zygo_dat_file.dat'
    mtfvfvf = 'valid_sample_MTFvFvF_Sag.txt'
    mtfvf = 'valid_sample_trioptics_mtf_vs_field.mht'
    mtf = 'valid_sample_trioptics_mtf.mht'

    def __call__(self, dtype_or_filename):
        """Get the path of a sample file."""
        dtype_or_filename = dtype_or_filename.lower()
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
