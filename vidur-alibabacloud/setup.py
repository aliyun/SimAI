from setuptools import find_packages, setup


setup(
    author="MSR-India Systems Group; Systems for AI Lab, Georgia Tech",
    python_requires='>=3.10',
    description="A LLM inference cluster simulator",
    include_package_data=True,
    keywords='vidur',
    name='vidur',
    packages=find_packages(include=['vidur', 'vidur.*']),
    version='0.0.1',
)
