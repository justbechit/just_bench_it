from just_bench_it.benchmarker import benchmark, run_benchmark
from just_bench_it.utils import print_results
import os

def set_github_token(token):
    os.environ['GITHUB_TOKEN'] = token
