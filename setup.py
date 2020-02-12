import setuptools

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ['tensorflow>=1.13.0',
                'pandas',
                'sklearn',
                    ]
    
setuptools.setup(
    name='hivae',
    version='0.16',
    url='https://github.com/gkoutos-group/hivae/',
    license='MIT',
    author='Andreas Karwath',
    author_email='a.karwath@bham.ac.uk',
    description='HIVAE (https://arxiv.org/pdf/1807.03653.pdf - by Nazabal, Olmos, Ghahramani, Valera) - extenstion of their implementations as Python library',
    packages=setuptools.find_packages(exclude=['examples']),
    install_requires=requirements,
    long_description=readme,
    long_description_content_type="text/markdown",
    zip_safe=False
    )


