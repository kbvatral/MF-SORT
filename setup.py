import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mf-sort",
    version="0.2",
    author="Caleb Vatral",
    author_email="caleb.m.vatral@vanderbilt.edu",
    description="Unofficial implementation of the Motion Feature SORT algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kbvatral/MF-SORT",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3.0 License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy',
          'scipy',
          'filterpy',
      ],
    extras_require={
        "yolo": ["yolov5", "imutils", "tqdm"],
    },
    python_requires='>=3.5',
)