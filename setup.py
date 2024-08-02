from setuptools import setup, find_packages

# Read requirements from file
with open('just_bench_it/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="just_bench_it",
    version="0.1.9",  # 初始版本号
    packages=find_packages(),
    install_requires=requirements,
    author="stone91",
    zip_safe=False,
    author_email="370025263@qq.com",
    description="A simple benchmarking tool for RL algorithms on Atari games",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/justbechit/just_bench_it",
)

