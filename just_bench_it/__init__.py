from just_bench_it.benchmarker import benchmark
from just_bench_it.utils import print_results
import os

__version__ = "0.1.8"
def set_github_token(token):
    os.environ['GITHUB_TOKEN'] = token

