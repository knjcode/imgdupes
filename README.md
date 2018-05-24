# imgdupes

![video_capture](video_capture.gif)
Images by [Caltech 101] dataset that semi-deduped for demonstration.

`imgdupes` is a command line tool for finding and deleting duplicate image files based on perceptual hash.  
You can delete duplicate image files with an operation similar to the [`fdupes`] command.  
It is better to pre-deduplicate identical files with [`fdupes`] in advance.



# Install

To install, simply use pip:

```bash
$ pip install imgdupes
```


# Usage

Find a set of images with Hamming distance of phash less than 4.
(To search images recursively from the target directory, add `-r` or `--recursive` option.)

```bash
$ imgdupes --recursive target_dir phash 4
```

The set of images are sorted in ascending order of file size and displayed together with the pixel size of the image, you choose which image to preserve.

If you are using iTerm 2, you can display a set of images on the terminal with the `-c` or `--imgcat` option.

```bash
$ imgdupes --recursive --imgcat 101_ObjectCategories phash 4
```


# Available hash algorithm

`imgdupes` uses the [ImageHash] to calculate perceptual hash.
You can use the same hash method as [ImageHash] library.

- [aHash]: average hashing
- [pHash]: perception hashing
- [dHash]: difference hashing
- [wHash]: wavelet hashing


# Options

`-r` `--recursive` (default `False`)

search images recursively from the target directory

`-c` `--imgcat` (default=False)

display duplicate images for iTerm2

`--log`

output logs of duplicate and delete files

`--size 256x256`

resize image (default=256x256)

`--space 0`

space between images (default=0)

`--space-color black`

space color between images (default=black)

`--tile-num 8`

space color between images (default=black)

`--no-keep-aspect`

do not keep aspect when displaying images

`--no-subdir-warning`

stop warnings that appear when similar images are in different subdirectories

`--no-cache`

do not create image hash cache

`--dry-run`

dry run (do not delete any files)


# License

MIT

[`fdupes`]: (https://github.com/adrianlopezroche/fdupes)
[Caltech 101]: http://www.vision.caltech.edu/Image_Datasets/Caltech101/
[ImageHash]: https://github.com/JohannesBuchner/imagehash
[aHash]: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
[pHash]: http://www.hackerfactor.com/blog/index.php?/archives/432-Looks-Like-It.html
[dHash]: http://www.hackerfactor.com/blog/index.php?/archives/529-Kind-of-Like-That.html
[wHash]: https://fullstackml.com/2016/07/02/wavelet-image-hash-in-python/
