"""Sample data for prysm tests and documentation."""
from pathlib import Path
from urllib.request import urlretrieve

baseremote = Path(r'https://github.com/brandondube/prysm/sampledata')
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
            path_ = (root / getattr(self, dtype)).absolute()
            if not path_.exists():
                urlretrieve(baseremote / filename, path_)

            return path_.absolute()

        else:
            raise ValueError('invalid sample filetype requested.')


sample_files = SampleFiles()
