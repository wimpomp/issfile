import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='issfile',
    version='2022.10.1',
    author='Wim Pomp @ Lenstra lab NKI',
    author_email='w.pomp@nki.nl',
    description='Open ISS files.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/wimpomp/issfile',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=['numpy', 'tqdm', 'pyyaml', 'tiffwrite>=2022.10.1'],
    entry_points={'console_scripts': ['iss2tiff=issfile:main']}
)
