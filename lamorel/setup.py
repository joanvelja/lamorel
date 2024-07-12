from setuptools import find_packages, setup

setup(
    name="lamorel",
    packages=find_packages("src"),
    package_dir={"": "src"},
    version="0.2",
    install_requires=[
        "transformers>=4.35",
        "accelerate>=0.24.1",
        "hydra-core",
        "torch>=2.1.0",
        "tqdm",
        "peft",
        "datasets",
        "huggingface_hub",
        "wandb",
        "bitsandbytes>=0.41.1",
    ],
    description="",
    author="Clément Romac (Hugging Face & Inria) - Patched by me",
)
