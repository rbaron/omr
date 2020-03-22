# omr

![Python application](https://github.com/rbaron/omr/workflows/omr/badge.svg)

omr is a small Python 3 **o**ptical **m**ark **r**ecognition script. It takes as input an image of an answered answer sheet and outputs which alternatives were marked. The project is generally optimized for conciseness and teachability, and the goal is to provide a reasonable starting point for learning and hopefully building more powerful applications.

## Usage
```sh
$ python omr.py --help
usage: omr.py [-h] --input INPUT [--output OUTPUT] [--show]

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    Input image filename
  --output OUTPUT  Output annotated image filename
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

## Installation
### Using virtualenv
```sh
$ git clone https://github.com/rbaron/omr
$ cd omr/
$ virtualenv --python=`which python3` venv
(venv) $ pip install -r requirements.txt
(venv) $ python omr.py --help
```

## Anwer Sheet
The answer sheet is available in the `sheet/` directory.

## Development
### Run unit tests
`test_omr.py` contains unit tests that can be run using:

```bash
$ py.test
=========================================================== test session starts ===========================================================
platform darwin -- Python 3.7.5, pytest-5.2.4, py-1.8.0, pluggy-0.13.0
collected 1 item

test_omr.py .                                                                                                                       [100%]

============================================================ 1 passed in 0.31s ============================================================
```

