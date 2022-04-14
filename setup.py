from setuptools import setup, find_packages
from setuptools.command.install import install

setup(name='cogmtc',
      packages=find_packages(),
      version="0.1.0",
      description='A project to determine how grounded language affects some cognitive counting tasks',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/cogmtc.git',
      install_requires= ["numpy",
                         "torch",
                         "tqdm"],
      py_modules=['supervised_gym'],
      long_description='''
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
