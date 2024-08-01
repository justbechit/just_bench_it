from just_bench_it.benchmarker import benchmark
from just_bench_it.utils import print_results
import os

__version__ = "0.1.5"
def set_github_token(token):
    os.environ['GITHUB_TOKEN'] = token

from . import _version
__version__ = _version.get_versions()['version']
