import argparse
import cv2
import math
import numpy as np

# When the four corners are identified, we will do a four-point
# perspective transform such that the outmost points of each
# corner maps to a TRANSF_SIZE x TRANSF_SIZE square image.
TRANSF_SIZE = 512

N_QUESTIONS = 10

# Answer sheet geometry in pixels
ANSWER_SHEET_WIDTH = 740
ANSWER_SHEET_HEIGHT = 1049

ANSWER_PATCH_HEIGHT = 50
ANSWER_PATCH_HEIGHT_WITH_MARGIN = 80
ANSWER_PATCH_LEFT_MARGIN = 200
ANSWER_PATCH_RIGHT_MARGIN = 90
FIRST_ANSWER_PATCH_TOP_Y = 200

ALTERNATIVE_HEIGHT = 50
ALTERNATIVE_WIDTH = 50
ALTERNATIVE_WIDTH_WITH_MARGIN = 100

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
    """Converts image to black and white.

    Applying a threshold to a grayscale image will make every pixel either
    fully black or fully white. Before doing so, a common technique is to
    get rid of noise (or super high frequency color change) by blurring the
    grayscale image with a Gaussian filter."""
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Filter the grayscale image with a 3x3 kernel
    blurred = cv2.GaussianBlur(im_gray, (3, 3), 0)

    # Applies a Gaussian adaptive thresholding. In practive, adaptive thresholding
    # seem to work better than appling a single, global threshold to the image.
    # This is particularly important if there could be shadows or non-uniform
    # lighting on the answer sheet. In those scenarios, using a global thresholding
    # technique might yield paricularly bad results.
    # The choice of the parameters blockSize = 77 and C = 10 is as much as an art
    # as a science and domain-dependand.
    # In practice, you might want to try different  values for your specific answer
    # sheet.
    return cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 77, 10)


def get_approx_contour(contour, tol=.01):
    """Get rid of 'useless' points in the contour"""
    epsilon = tol * cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, epsilon, True)


def get_contours(image_gray):
    contours, hierarchy = cv2.findContours(
        image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return map(get_approx_contour, contours)


def get_corners(contours):
    """Return the 4 contours that most look like a corner.

    In practice, we cannot assume that the corners will always be present,
    and we likely need to decide how good is good enough for contour to
    look like a corner.
    This is essentially a classification problem. A good approach would be
    to train a statistical classifier model and apply it here. For our
    exercise, we assume the corners are necessarily there."""
    corner_features = calculate_corner_features()
    return sorted(
        contours,
        key=lambda c: features_distance(
                corner_features,
                calculate_contour_features(c)))[:4]


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
        TRANSF_SIZE * x / ANSWER_SHEET_WIDTH,
        TRANSF_SIZE * y / ANSWER_SHEET_HEIGHT
    )))


def get_question_patch(transf, question_index):
    """Exracts a region of interest (ROI) of a single question."""
    # Top left of question patch q_number
    tl = sheet_coord_to_transf_coord(
        ANSWER_PATCH_LEFT_MARGIN,
        FIRST_ANSWER_PATCH_TOP_Y + ANSWER_PATCH_HEIGHT_WITH_MARGIN * question_index
    )

    # Bottom right of question patch q_number
    br = sheet_coord_to_transf_coord(
        ANSWER_SHEET_WIDTH - ANSWER_PATCH_RIGHT_MARGIN,
        FIRST_ANSWER_PATCH_TOP_Y +
            ANSWER_PATCH_HEIGHT +
            ANSWER_PATCH_HEIGHT_WITH_MARGIN * question_index
    )
    return transf[tl[1]:br[1], tl[0]:br[0]]


def get_question_patches(transf):
    for i in range(N_QUESTIONS):
        yield get_question_patch(transf, i)


def get_alternative_patches(question_patch):
    for i in range(5):
        x0, _ = sheet_coord_to_transf_coord(ALTERNATIVE_WIDTH_WITH_MARGIN * i, 0)
        x1, _ = sheet_coord_to_transf_coord(ALTERNATIVE_WIDTH + ALTERNATIVE_WIDTH_WITH_MARGIN * i, 0)
        yield question_patch[:, x0:x1]


def draw_marked_alternative(question_patch, index):
    cx, cy = sheet_coord_to_transf_coord(
        ALTERNATIVE_WIDTH * (2 * index + .5),
        ALTERNATIVE_HEIGHT / 2)
    draw_point((cx, cy), question_patch, radius=5, color=(255, 0, 0))


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
        - Normalize image
        - Find contours
        - Find corners among all contours
        - Find 'outmost' points of all corners
        - Apply perpsective transform to get a bird's eye view
        - Scan each line for the marked alternative
    """

    corner_features = calculate_corner_features()

    im_orig = cv2.imread(source_file)

    im_normalized = normalize(im_orig)

    contours = get_contours(im_normalized)

    corners = get_corners(contours)

    cv2.drawContours(im_orig, corners, -1, (0, 255, 0), 3)

    outmost = order_points(get_outmost_points(corners))

    color_transf = perspective_transform(im_orig, outmost)
    normalized_transf = perspective_transform(im_normalized, outmost)

    answers = []
    for i, q_patch in enumerate(get_question_patches(normalized_transf)):
        alt_index = get_marked_alternative(get_alternative_patches(q_patch))

        if alt_index is not None:
            color_q_patch = get_question_patch(color_transf, i)
            draw_marked_alternative(color_q_patch, alt_index)

        answers.append(get_letter(alt_index))

    return answers, color_transf


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
