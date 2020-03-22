import argparse
import cv2
import math
import numpy as np


TRANSF_SIZE = 512


def calculate_contour_features(contour):
    """Calculates interesting properties (features) of a contour.

    We use these features to match shapes (contours). In this script,
    we are interested in finding shapes in our input image that look like
    a corner. We do that by calculating the features for many contours
    in the input image and comparing these to the features of the corner
    contour. By design, we know exactly what the features of the real corner
    contour look like - check out the calculate_corner_features function.

    It is crucial for these features to be invariant both to scale and rotation.
    In other words, we know that a corner is a corner regardless of its size
    or rotation. In the past, this script implemented its own features, but
    OpenCV offers much more robust scale and rotational invariant features
    out of the box - the Hu moments.
    """
    moments = cv2.moments(contour)
    return cv2.HuMoments(moments)


def calculate_corner_features():
    """Calculates the array of features for the corner contour.
    In practive, this can be pre-calculated, as the corners are constant
    and independent from the inputs.

    We load the img/corner.png file, which contains a single corner, so we
    can reliably extract its features. We will use these features to look for
    contours in our input image that look like a corner.
    """
    corner_img = cv2.imread('img/corner.png')
    corner_img_gray = cv2.cvtColor(corner_img, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(
        corner_img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # We expect to see only two contours:
    # - The "outer" contour, which wraps the whole image, at hierarchy level 0
    # - The corner contour, which we are looking for, at hierarchy level 1
    # If in trouble, one way to check what's happening is to draw the found contours
    # with cv2.drawContours(corner_img, contours, -1, (255, 0, 0)) and try and find
    # the correct corner contour by drawing one contour at a time. Ideally, this
    # would not be done at runtime.
    if len(contours) != 2:
        raise RuntimeError(
            'Did not find the expected contours when looking for the corner')

    # Following our assumptions as stated above, we take the contour that has a parent
    # contour (that is, it is _not_ the outer contour) to be the corner contour.
    # If in trouble, verify that this contour is the corner contour with
    # cv2.drawContours(corner_img, [corner_contour], -1, (255, 0, 0))
    corner_contour = next(ct for i, ct in enumerate(contours)
                             if hierarchy[0][i][3] != -1)

    return calculate_contour_features(corner_contour)


def normalize(im):
    return cv2.normalize(im, np.zeros(im.shape), 0, 255, norm_type=cv2.NORM_MINMAX)


def get_approx_contour(contour, tol=.01):
    """Get rid of 'useless' points in the contour"""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_contours(image_gray):
    contours, hierarchy = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return map(get_approx_contour, contours)


def get_corners(contours):
    CORNER_FEATS = calculate_corner_features()
    return sorted(
        contours,
        key=lambda c: features_distance(CORNER_FEATS, calculate_contour_features(c)))[:4]


def get_bounding_rect(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)


def features_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))


def draw_point(point, img, radius=5, color=(0, 0, 255)):
    cv2.circle(img, tuple(point), radius, color, -1)


def get_centroid(contour):
    m = cv2.moments(contour)
    x = int(m["m10"] / m["m00"])
    y = int(m["m01"] / m["m00"])
    return (x, y)


def order_points(points):
    """Order points counter-clockwise-ly."""
    origin = np.mean(points, axis=0)

    def positive_angle(p):
        x, y = p - origin
        ang = np.arctan2(y, x)
        return 2 * np.pi + ang if ang < 0 else ang

    return sorted(points, key=positive_angle)


def get_outmost_points(contours):
    all_points = np.concatenate(contours)
    return get_bounding_rect(all_points)


def perspective_transform(img, points):
    """Transform img so that points are the new corners"""
    source = np.array(
        points,
        dtype="float32")

    dest = np.array([
        [TRANSF_SIZE, TRANSF_SIZE],
        [0, TRANSF_SIZE],
        [0, 0],
        [TRANSF_SIZE, 0]],
        dtype="float32")

    img_dest = img.copy()
    transf = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(img, transf, (TRANSF_SIZE, TRANSF_SIZE))
    return warped


def sheet_coord_to_transf_coord(x, y):
    return list(map(lambda n: int(np.round(n)), (
        TRANSF_SIZE * x/744.055,
        TRANSF_SIZE * (1 - y/1052.362)
    )))


def get_question_patch(transf, q_number):
    # Top left
    tl = sheet_coord_to_transf_coord(
        200,
        850 - 80 * (q_number - 1)
    )

    # Bottom right
    br = sheet_coord_to_transf_coord(
        650,
        800 - 80 * (q_number - 1)
    )
    return transf[tl[1]:br[1], tl[0]:br[0]]


def get_question_patches(transf):
    for i in range(1, 11):
        yield get_question_patch(transf, i)


def get_alternative_patches(question_patch):
    for i in range(5):
        x0, _ = sheet_coord_to_transf_coord(100 * i, 0)
        x1, _ = sheet_coord_to_transf_coord(50 + 100 * i, 0)
        yield question_patch[:, x0:x1]


def draw_marked_alternative(question_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        50 * (2 * index + .5),
        50/2)
    draw_point((cx, TRANSF_SIZE - cy), question_patch, radius=5, color=(255, 0, 0))


def get_marked_alternative(alternative_patches):
    means = list(map(np.mean, alternative_patches))
    sorted_means = sorted(means)

    # Simple heuristic
    if sorted_means[0]/sorted_means[1] > .7:
        return None

    return np.argmin(means)


def get_letter(alt_index):
    return ["A", "B", "C", "D", "E"][alt_index] if alt_index is not None else "N/A"


def get_answers(source_file):
    """Run the full pipeline:

        - Load image
        - Convert to grayscale
        - Filter out high frequencies with a Gaussian kernel
        - Apply threshold
        - Find contours
        - Find corners among all contours
        - Find 'outmost' points of all corners
        - Apply perpsective transform to get a bird's eye view
        - Scan each line for the marked answer
    """

    corner_features = calculate_corner_features()

    im_orig = cv2.imread(source_file)

    blurred = cv2.GaussianBlur(im_orig, (11, 11), 10)

    im = normalize(cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY))

    ret, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

    contours = get_contours(im)

    corners = get_corners(contours)

    cv2.drawContours(im_orig, corners, -1, (0, 255, 0), 3)

    outmost = order_points(get_outmost_points(corners))

    transf = perspective_transform(im_orig, outmost)

    answers = []
    for i, q_patch in enumerate(get_question_patches(transf)):
        alt_index = get_marked_alternative(get_alternative_patches(q_patch))

        if alt_index is not None:
            draw_marked_alternative(q_patch, alt_index)

        answers.append(get_letter(alt_index))

    return answers, transf


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        help="Input image filename",
        required=True,
        type=str)

    parser.add_argument(
        "--output",
        help="Output image filename",
        type=str)

    parser.add_argument(
        "--show",
        action="store_true",
        help="Displays annotated image")

    args = parser.parse_args()

    answers, im = get_answers(args.input)

    for i, answer in enumerate(answers):
        print("Q{}: {}".format(i + 1, answer))

    if args.output:
        cv2.imwrite(args.output, im)
        print('Wrote image to {}.'.format(args.output))

    if args.show:
        cv2.imshow('Annotated image', im)
        cv2.waitKey(0)


if __name__ == '__main__':
    main()
