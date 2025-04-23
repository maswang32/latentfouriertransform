from setuptools import setup, find_packages

setup(
    name="fmdiffae",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "torch",
        "torchaudio",
        "torchvggish",
        "bigvgan @ git+https://github.com/maswang32/BigVGAN.git",
    ],
)
