import setuptools

setuptools.setup(
    name="imgdupes",
    version="0.0.6",
    author="Kenji Doi",
    author_email="knjcode@gmail.com",
    description="CLI tool to dedupe images based on perceptual hash",
    long_description="Finding and deleting duplicate image files based on perceptual hash.",
    license="MIT",
    keywords="image dedupe perceptual hash",
    url="https://github.com/knjcode/imgdupes",
    packages=setuptools.find_packages(),
    scripts=['imgdupes'],
    install_requires=[
        'future',
        'ImageHash',
        'joblib',
        'opencv-python',
        'pathlib',
        'Pillow',
        'six',
        'termcolor',
        'tqdm',
        'webcolors',
    ],
)
