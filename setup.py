from setuptools import setup, find_packages



# ensure that torch is installed, and send to torch website if not
try:
    import torch
except ModuleNotFoundError:
    raise ValueError("Please install torch first: https://pytorch.org/get-started/locally/")

_REQUIRED = [
    "protobuf<4.24",
    "fsspec==2023.10.0",
    "datasets==2.15.0",
    "aiohttp", # https://github.com/aio-libs/aiohttp/issues/6794
    "dill==0.3.6",
    "multiprocess==0.70.14",
    "huggingface-hub==0.23.4",
    "transformers==4.42.3",
    "einops==0.7.0",
    "ftfy==6.1.3",
    "opt-einsum==3.3.0",
    "pydantic==2.5.3",
    "pydantic-core==2.14.6",
    "pykeops==2.2",
    "python-dotenv==1.0.0",
    "sentencepiece==0.1.99",
    "six==1.16.0",
    "flash-attn==2.5.2",
    "mamba_ssm==2.0.4",
    "rich",
    "hydra-core==1.3.2",
    "hydra_colorlog",
    "wandb==0.16.2",
    "ray==2.24.0",
    "sacrebleu",
    "causal-conv1d",
    # "scikit-learn==1.3.2",
    # "lm-eval==0.4.1",
    # "ninja==1.11.1.1",
]

_OPTIONAL = {
    "train": [
        "lightning-bolts==0.7.0",
        "lightning-utilities==0.10.0",
        "pytorch-lightning==1.8.6",
        "timm"
    ],
    "dev": [
        "pytest"
    ]
}


setup(
    name='jrt',
    version="0.0.1",
    packages=find_packages(include=['based', 'based.*']),
    author="JRT",
    author_email="",
    description="",
    python_requires=">=3.8",
    install_requires=_REQUIRED,
    extras_require=_OPTIONAL,
)
