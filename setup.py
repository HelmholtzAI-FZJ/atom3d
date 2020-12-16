from setuptools import setup, find_packages

setup(
    name='atom3d',
    packages=find_packages(include=[
        'atom3d',
        'atom3d.protein',
        'atom3d.shard',
        'atom3d.util',
        'atom3d.datasets',
    ]),
    version='0.1.0',
    description='ATOM3D: Tasks On Molecules in 3 Dimensions',
    author='Raphael Townshend',
    license='MIT',
    install_requires=[
        'argparse',
        'biopython',
        'click',
        'easy-parallel',
        'h5py',
        'lmdb',
        'msgpack',
        'numpy',
        'pandas',
        'python-dotenv>=0.5.1',
        'scipy',
        'tables',
        'torch',
        'tqdm',
    ],
)
