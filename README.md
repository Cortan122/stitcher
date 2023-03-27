# Screenshot Stitcher

The default OpenCV image stitcher is designed to work specifically with panoramic photographs, and usually fails if you try to stitch a lot of similar screenshots.
I was trying to stitch some Miro screenshots, and could not find an easy way to do that...
So here is my own solution!!

## Usage

```bash
./vert_stacker.py [files] outfile.png
```

You might also have to modify some of the constants defined at the top of the python file.

## Presentation (in Russian)

Презентация: https://docs.google.com/presentation/d/1Hv86JdFjxEzTlkIZ0UJX1rIEY-Isgz2uR84UT-XG9p4 \
Код: https://colab.research.google.com/drive/1WkktICAHZhLfWKej0pKsa1_7aPKbCGms

## Todo list

- [ ] Use a neural network as a custom Feature detector
- [ ] Use tesseract OCR as a custom Feature detector
- [ ] Parse argv options
- [ ] Improve repeated UI detection
