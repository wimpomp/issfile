import os
import setuptools

version = '2022.11.0'

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open(os.path.join(os.path.dirname(__file__), 'issfile', '_version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")
    try:
        with open(os.path.join(os.path.dirname(__file__), '.git', 'HEAD')) as g:
            head = g.read().split(':')[1].strip()
        with open(os.path.join(os.path.dirname(__file__), '.git', head)) as h:
            f.write("__git_commit_hash__ = '{}'\n".format(h.read().rstrip('\n')))
    except Exception:
        f.write(f"__git_commit_hash__ = 'unknown'\n")

setuptools.setup(
    name='issfile',
    version=version,
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
    install_requires=['numpy', 'matplotlib', 'tqdm', 'pyyaml', 'tiffwrite>=2022.10.2'],
    entry_points={'console_scripts': ['iss2tiff=issfile:main']}
)
