#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
from setuptools import setup, find_packages, Extension

try:
    from Cython.Build import cythonize

    ext_modules = cythonize([
        Extension("LinguisticTokenizer.tokenizer", ["LinguisticTokenizer/tokenizer.pyx"]),
    ])
except ImportError:
    ext_modules = None

setup(
    name='LinguisticTokenizer',
    version='0.0.1',
    packages=find_packages(),
    description='Linguistic Tokenizer',
    ext_modules=ext_modules,
    package_data={'LinguisticTokenizer': ['resources/*']}
)




