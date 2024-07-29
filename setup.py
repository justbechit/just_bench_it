from setuptools import setup, find_packages
import just_bench_it


setup(
    name="just_bench_it",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium[atari]",
        "numpy",
    ],
    author="Your Name",
    zip_safe=False,
    author_email="your.email@example.com",
    description="A simple benchmarking tool for RL algorithms on Atari games",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/just_bench_it",
)
