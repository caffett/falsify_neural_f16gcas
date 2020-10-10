import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="F16-Falsify", # Replace with your own username
    version="0.0.1",
    author="Zikang Xiong",
    author_email="zikangxiong@gmail.com",
    description="Bayesian Optimization (BO) based falsify on F16 with neural network controller",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/caffett/f16-falsify",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)