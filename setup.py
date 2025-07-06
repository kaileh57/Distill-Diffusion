from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="diffusion-llm-converter",
    version="0.1.0",
    author="Diffusion LLM Converter Team",
    author_email="contact@example.com",
    description="Convert pre-trained autoregressive language models into diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/diffusion-llm-converter",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "black", "flake8", "mypy"],
        "gpu": ["torch[cuda]", "flash-attn"],
    },
    entry_points={
        "console_scripts": [
            "diffusion-convert=scripts.train_diffusion:main",
            "diffusion-eval=scripts.evaluate:main",
            "diffusion-download=scripts.download_model:main",
        ],
    },
)