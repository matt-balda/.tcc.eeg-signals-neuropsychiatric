from setuptools import setup, find_packages

setup(
    name='balda-neurokitx',
    version='0.0.1',
    author='Mateus Balda Mota',
    author_email='mateusbalda89@gmail.com',
    description='Extensible package for neurophysiological data manipulation with an emphasis on mental disorders',
    #long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='',
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        'pandas'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
