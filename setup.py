from setuptools import setup, find_packages

packages=find_packages()  # Automatically discover and include all packages
print(packages)
setup(
    name='long-timescale-analysis',
    version='0.1.0',
    description='A short description of your package',
    author='foo',
    author_email='foo@email.com',
    packages=packages,
    url='https://github.com/shaevitz-lab/long-timescale-analysis',
    install_requires=[
        "numpy",
        "natsort",
        # "awkde" pip install git+https://github.com/mennthor/awkde
        "palettable",
        "codetiming",
        "hdf5storage",
        
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
