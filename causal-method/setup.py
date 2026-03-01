from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Define the extensions with paths relative to the project root
extensions = [
    Extension(
        "model.cy_utils", 
        ["src/model/cy_utils.pyx"], 
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "model.utils", 
        ["src/model/utils.py"], 
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "model.sequence_extractor", 
        ["src/model/sequence_extractor.py"], 
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "model.cy_sequence_extractor", 
        ["src/model/cy_sequence_extractor.pyx"], 
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    ext_modules=cythonize(extensions, language_level="3", annotate=True)
)