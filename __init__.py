from .benchmarker import benchmark, run_benchmark
import os

def set_github_token(token):
    os.environ['GITHUB_TOKEN'] = token
