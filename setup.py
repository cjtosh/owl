from setuptools import setup

setup(
    name='owl',
    version='1.0',
    description='Optimistic weighted likelihood for misspecified probabilistic models.',
    author='Christopher Tosh',
    author_email='christopher.j.tosh@gmail.com',
    url='https://github.com/cjtosh/owl',
    license="GNU GPLv3",
    packages=['owl'],
    install_requires=['numpy', 'scipy', 'sklearn', 'tqdm', 'kneed', 'pandas'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python'
        ],
    keywords=['probabilistic-modeling', 'robust-statistics','maximum-likelihood'],
    platforms="ALL"
)