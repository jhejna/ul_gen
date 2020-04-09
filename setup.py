import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

print(setuptools.find_packages())

setuptools.setup(
    name="ul_gen", 
    version="0.0.1",
    author="Ashwin Vangipuram Joey Hejna, Kara Liu",
    description="Code for our Deep UL procedural generation project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    requires=['rlpyt']
)

