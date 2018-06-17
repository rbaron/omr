# omr

omr is a simple python2 **o**ptical **m**ark **r**ecognition script. It takes as input an image of an answered answer sheet and outputs which alternatives were marked. Scroll down for an example.

## Usage
```sh
$ python2 omr.py --help
usage: omr.py [-h] --input INPUT [--output OUTPUT] [--show]

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Input image filename
  --output OUTPUT  Output image filename
  --show           Displays annotated image
```

## Example

```sh
$ python omr.py --input img/answered-sheet-photo.jpg  --output /tmp/results.png --show

Q1: A
Q2: C
Q3: C
Q4: E
Q5: N/A
Q6: N/A
Q7: A
Q8: N/A
Q9: N/A
Q10: N/A
```

In this case, we used the following image as input:

<img src="http://i.imgur.com/JTAgYNF.jpg" alt="Input" style="max-width: 50%;"/>

And got the following output:

<img src="http://i.imgur.com/4n9fKFF.png" alt="Output" style="max-width: 50%;"/>

## Anwer Sheet

The answer sheet is available in the `sheet/` directory.

## Development

### Run unit tests

`test_omr.py` contains unit tests that can be run using:

```bash
$ py.test
======================================================== test session starts =========================================================
platform darwin -- Python 3.6.5, pytest-3.5.1, py-1.5.3, pluggy-0.6.0
rootdir: /Users/user/omr, inifile:
plugins: remotedata-0.2.1, openfiles-0.3.0, doctestplus-0.1.3, arraydiff-0.2
collected 1 item

test_omr.py .                                                                                                                  [100%]

====================================================== 1 passed in 1.61 seconds ======================================================

```
