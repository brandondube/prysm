"""Sample data for prysm tests and documentation."""
import shutil
from pathlib import Path
from urllib.request import urlopen

baseremote = r'https://github.com/brandondube/prysm/raw/master/sample_files/'
baselocal = Path(__file__)
root = baselocal.parent / 'sampledata'
root.mkdir(exist_ok=True)


class SampleFiles:
    """Sample files for prysm."""
    dat = 'valid_zygo_dat_file.dat'
    mtfvfvf = 'valid_sample_MTFvFvF_Sag.txt'
    mtfvf = 'valid_sample_trioptics_mtf_vs_field.mht'

    def __call__(self, dtype):
        """Get the path of a sample file."""
        dtype = dtype.lower()
        if hasattr(self, dtype):
            filename = getattr(self, dtype)
            local = (root / getattr(self, dtype)).absolute()
            remote = baseremote + filename
            if not local.exists():
                with urlopen(remote) as response, open(local, 'wb') as fid:
                    shutil.copyfileobj(response, fid)

            return local

        else:
            raise ValueError('invalid sample filetype requested.')


sample_files = SampleFiles()
