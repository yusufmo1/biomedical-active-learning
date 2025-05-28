from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="biomedical-active-learning",
    version="0.1.0",
    author="Yusuf Mohammed",
    author_email="your.email@example.com",
    description="Active Learning for Biomedical Data: Superior Performance with Minimal Labels",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/biomedical-active-learning",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0"],
        "notebooks": ["jupyter>=1.0.0", "notebook>=7.0.0", "ipywidgets>=8.1.0"],
    },
    entry_points={
        "console_scripts": [
            "biomedical-al=src.main:main",
        ],
    },
)