from setuptools import setup, find_packages

setup(
    name='packages-mental-disorder-eeg-data',
    version='0.0.4',
    author='Mateus Balda Mota',
    author_email='mateusbalda89@gmail.com',
    description='Pacote para manipular conjuntos',
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
