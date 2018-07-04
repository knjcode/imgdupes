# imgdupes

`imgdupes` is a command line tool for finding and deleting duplicate and/or similar image files based on perceptual hash.

![video_capture](video_capture.gif)
Images by [Caltech 101] dataset that semi-deduped for demonstration.

You can delete duplicate image files with an operation similar to the [`fdupes`] command.  
It is better to pre-deduplicate identical files with [`fdupes`] in advance.


## For large dataset

It is possible to speed up dedupe process by approximate nearest neighbor search of hamming distance with [NGT] or [hnsw].
See [Against large dataset](#against-large-dataset) section for details.


# Install

To install, simply use pip:

```bash
$ pip install imgdupes
```


# Usage

Find a set of images with Hamming distance of phash less than 4 from target directory.  
To search images recursively from the target directory, add `-r` or `--recursive` option.

```bash
$ imgdupes --recursive <target_dir> phash 4
target_dir/airplane_0583.jpg
target_dir/airplane_0800.jpg

target_dir/watch_0122.jpg
target_dir/watch_0121.jpg
```

By default, imgdupes displays a list of duplicate images list and exits.  
To display preserve or delete images prompt, use the `-d` or `--delete` option.

If you are using iTerm 2, you can display a set of images on the terminal with the `-c` or `--imgcat` option.

```bash
$ imgdupes --recursive --delete --imgcat 101_ObjectCategories phash 4
```

The set of images are sorted in ascending order of file size and displayed together with the pixel size of the image, you choose which image to preserve.


# Against large dataset

`imgdupes` supports approximate nearest neighbor search of hamming distance with [NGT] or [hnsw].

To dedupe images using NGT, run with `--ngt` option after installing NGT and python binding.

```bash
$ imgdupes -rdc --ngt 101_ObjectCategories phash 4
```

For instructions on installing NGT and python binding, see [NGT] and [python NGT].

To dedupe images using hnsw, run with `--hnsw` option after installing hnsw python binding.

```bash
$ imgdupes -rdc --hnsw 101_ObjectCategories phash 4
```


# Fast exact searching

`imgdupes` supports exact nearest neighbor search of hamming distance with [faiss] (IndexFlatL2).

To dedupe images using faiss, run with `--faiss-flat` option after installing faiss python binding.

```bash
$ imgdupes -rdc --faiss-flat 101_ObjectCategories phash 4
```


# Using imgdupes without installing it with docker

You can use `imgdupes` without installing it using a pre-build docker container image.  
[NGT], [hnsw] and [faiss] are already installed in this image.

Place the target directory in the current directory and execute the following command.

```bash
$ docker run -it -v $PWD:/app knjcode/imgdupes -rdc --ngt <target_dir> phash 0
```

When docker run, current directory is mounted inside the container and referenced from imgdupes.


By aliasing the command, you can use `imgdupes` as installed.

```bash
$ alias imgdupes="docker run -it -v $PWD:/app knjcode/imgdupes"
$ imgdupes -rdc --hnsw <target_dir> phash 0
```


To upgrade imgdupes docker image, you can pull the docker image as below.

```bash
$ docker pull knjcode/imgdupes
```


# Available hash algorithm

`imgdupes` uses the [ImageHash] to calculate perceptual hash.
You can use the same hash method as [ImageHash] library.

- [ahash]: average hashing
- [phash]: perception hashing
- [dhash]: difference hashing
- [whash]: wavelet hashing


# Options

`-r` `--recursive`

search images recursively from the target directory (default=False)

`-d` `--delete`

prompt user for files to preserve and delete (default=False)

`-c` `--imgcat`

display duplicate images for iTerm2 (default=False)

`--hash-bits 64`

bits of perceptual hash (default=64)

The number of bits specifies the value that is the square of n.  
For example, you can specify 64(8^2), 144(12^2), 256(16^2), etc.

`--sort <sort_type>`

how to sort duplicate image files (default=filesize)

You can specify following types:

- `filesize`: sort by filesize in descending order
- `filepath`: sort by filepath in ascending order
- `imagesize`: sort by pixel width and height in descenging order
- `width`: sort by pixel width in descending order
- `height`: sort by pixel height in descending order

`--reverse`

reverse sort order

`--num-proc 4`

number of hash calculation and ngt processes (default=cpu_count-1)

`--log`

output logs of duplicate and delete files (default=False)

`--cache`

create and use image hash cache (default=False)

`--size 256x256`

resize image (default=256x256)

`--space 0`

space between images (default=0)

`--space-color black`

space color between images (default=black)

`--tile-num 4`

horizontal tile number (default=4)

`--interpolation INTER_LINEAR`

interpolation methods (default=INTER_LINEAR)

You can specify OpenCV interpolation methods: INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4, etc.

`--no-keep-aspect`

do not keep aspect when displaying images

`--no-subdir-warning`

stop warnings that appear when similar images are in different subdirectories

`--dry-run`

dry run (do not delete any files)

`--faiss-flat`

use faiss exact search (IndexFlatL2) for calculating Hamming distance between hash of images (default=False)


## ngt options

`--ngt`

use NGT for calculating Hamming distance between hash of images (default=False)

`--ngt-k 20`

number of searched objects when using NGT.
Increasing this value, improves accuracy and increases computation time. (default=20)

`--ngt-epsilon 0.1`

search range when using NGT.
Increasing this value, improves accuracy and increases computation time. (default=0.1)


## hnsw options

`--hnsw`

use hnsw for calculating Hamming distance between hash of images (default=False)

`--hnsw-k 20`

number of searched objects when using hnsw.
Increasing this value, improves accuracy and increases computation time. (default=20)

`--hnsw-ef-construction 100`

controls index search speed/build speed tradeoff (default=100)

`--hnsw-m 16`

m is tightly connected with internal dimensionality of the data stronlgy affects the memory consumption (default=16)

`--hnsw-ef 50`

controls recall. higher ef leads to better accuracy, but slower search (default=50)


# License

MIT

[`fdupes`]: (https://github.com/adrianlopezroche/fdupes)
[Caltech 101]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
[ImageHash]: https://github.com/JohannesBuchner/imagehash
[ahash]: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
[phash]: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
[dhash]: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
[whash]: https://fullstackml.com/2016/07/02/wavelet-image-hash-in-python/
[NGT]: https://github.com/yahoojapan/NGT
[python NGT]: https://github.com/yahoojapan/NGT/tree/master/python
[hnsw]: https://github.com/nmslib/hnsw
[faiss]: https://github.com/facebookresearch/faiss
