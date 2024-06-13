import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torch_vggish_yamnet",
    version="0.1.4",
    author="Stefano Giacomelli (Ph.D. student UnivAQ)",
    author_email="stefano.giacomelli@graduate.univaq.it",
    description="torch_vggish_yamnet: PyTorch VGGish & YAMNet models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StefanoGiacomelli/torch_vggish_yamnet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy', 'torch', 'torchaudio'],
    python_requires='>=3.6',
)
