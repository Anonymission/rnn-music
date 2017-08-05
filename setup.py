from distutils.core import setup
from setuptools import find_packages

try:
    import mido
except ImportError:
    print("Warning: `mido` must be installed in order to use `rnn_music`")

setup(name='rnn_music',
      version='1.0',
      description='Generates music',
      author='Petar Griggs (@Anonymission)',
      author_email="marrs2k@gmail.com",
      packages=find_packages(),
      license="MIT"
      )
