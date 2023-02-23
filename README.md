# issfile
Library for opening [ISS](https://iss.com) files and their conversion to tiff.

## Installation

    pip install issfile

## Converting .iss-pt files to .tiff files

    iss2tiff --help

    iss2tiff file.iss-pt

this will create file.tif and file.carpet.tif containing images and carpets respectively.
Metadata is also saved in the tiffs in the description tag. A pdf with some plots is created too.

## Use as a library

    from matplotlib import pyplot as plt
    from issfile import IssFile

    with IssFile(file) as iss:
        image = iss.get_image(c=1, t=5)
        carpet, (time, x, y, z) = iss.get_carpet(c=1, t=5)
    
        plt.figure()
        plt.imshow(image)
        plt.figure()
        plt.plot(carpet.sum(1))
