import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name="Last_Statement_Binary_Classification",
    version='0.0.1',
    author='DSAN 5400 Project Group',
    author_email='sg1951@georgetown.edu',
    description='Files used to perform TF-IDF weighting and logistic regression on text data to predict if the text was spoken by a labeled criminal or not.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    extras_require={"dev": ["pytest", "flake8", "autopep8"]},
)