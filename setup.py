import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    

__version__ = "0.0.0"

REPO_NAME = "medical-dcgan"
Author_USER = "Indrayani-B"
SRC_REPO = "dcGAN_image_generation"
Author_EMAIL = "indrayanibhujade378@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=Author_USER,
    author_email=Author_EMAIL,
    description="A small python package for DCGAN based medical image generation",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{Author_USER}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{Author_USER}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)