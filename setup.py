from setuptools import setup
from setuptools import find_packages

setup(name='speech2speech',
      version='0.1.0',
      description='Generating synthetic speech',
      url='https://github.com/bithikajain/speech2speech',
      author='Bithika Jain',
      author_email='jain.bithika01@gmail.com',
      license='MIT',
      packages=['speech2speech'],
      install_requires = [
      'numpy',
      'matplotlib',
      'librosa',
      'torch',
      'pandas',
      'torchviz',
      'torchsummary',
      'scipy'])
