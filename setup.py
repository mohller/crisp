from setuptools import setup

setup(
    name='crisp-py',
    version='0.1.0',
    packages=['crisp-py'],
    install_requires=[
        'astropy',
        'numpy',
        'pandas',
        'scipy',
        'pint',
        'sympy'        
    ],
    description='Cosmic Ray Stochastic Interactions for Propagation',
    long_description='A package to compute stochastic cosmic ray photonuclear interactions with probability distributions.',
    long_description_content_type='text/markdown',
    author='Leonel Morejon',
    author_email='leonel.morejon@uni-wuppertal.de',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
    ]
)
