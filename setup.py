from setuptools import setup, find_packages

setup(
    name="sudy",
    version="0.1.0",
    author="Sudy Team",
    description="Gelişmiş MoE ve MLA Mimarisine Sahip Verimli Türkçe LLM",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.3",
        "fastapi>=0.95.0",
        "uvicorn>=0.22.0",
        "pyyaml>=6.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "pydantic>=2.0.0",
        "requests>=2.28.0",
        "tqdm>=4.65.0",
        "pydantic-settings>=2.0.0"
    ],
    python_requires=">=3.8",
)
