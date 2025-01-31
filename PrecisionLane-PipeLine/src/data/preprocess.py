import cv2

def normalize_image(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

def reduce_noise(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


def preprocess(data):
    processed_data = []
    for image_path in data['image_paths']:
        image = cv2.imread(image_path)
        normalized_image = normalize_image(image)
        denoised_image = reduce_noise(normalized_image)
        processed_data.append(denoised_image)
    return processed_data
