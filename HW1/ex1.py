import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Load an image from file as function
def load_image(image_path):
    """
    Load an image from file, using OpenCV
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image
    pass

# Display an image as function
def display_image(image, title="Image"):
    """
    Display an image using matplotlib. Rembember to use plt.show() to display the image
    """
    plt.title(title)
    plot = plt.imshow(image)
    plt.show()
    pass


# grayscale an image as function
def grayscale_image(image):
    """
    Convert an image to grayscale. Convert the original image to a grayscale image. In a grayscale image, the pixel value of the
    3 channels will be the same for a particular X, Y coordinate. The equation for the pixel value
    [1] is given by:
        p = 0.299R + 0.587G + 0.114B
    Where the R, G, B are the values for each of the corresponding channels. We will do this by
    creating an array called img_gray with the same shape as img
    """
    R = np.array(image[:, :, 0])
    G = np.array(image[:, :, 1])
    B = np.array(image[:, :, 2])
    avg = R * .299 + G * .587 + B * .114
    for i in range (0, 3):
        image[:, :, i] = avg
    return image
    pass


# Save an image as function
def save_image(image, output_path):
    """
    Save an image to file using OpenCV
    """
    cv2.imwrite(output_path, image)
    pass


# flip an image as function 
def flip_image(image):
    """
    Flip an image horizontally using OpenCV
    """
    image = cv2.flip(src = image, flipCode = 1)
    return image
    pass


# rotate an image as function
def rotate_image(image, angle):
    """
    Rotate an image using OpenCV. The angle is in degrees
    """
    height, width = image.shape[:2]
    rotateMatrix = cv2.getRotationMatrix2D(center = (width / 2, height / 2),
                                            angle = angle, scale = 1)
    resultImage = cv2.warpAffine(src = image, M = rotateMatrix, dsize = (width, height))
    return resultImage
    pass


if __name__ == "__main__":
    # Load an image from file
    img = load_image("uet.png")

     # Display the image
    display_image(img, "Original Image")

    # Convert the image to grayscale
    img_gray = grayscale_image(img)

    # Display the grayscale image
    display_image(img_gray, "Grayscale Image")

    # Save the grayscale image
    save_image(img_gray, "images/lena_gray.jpg")

    # Flip the grayscale image
    img_gray_flipped = flip_image(img_gray)

    # Display the flipped grayscale image
    display_image(img_gray_flipped, "Flipped Grayscale Image")

    # Save the flipped image
    save_image(img_gray_flipped, "images/lena_gray_flipped.jpg")

    # Rotate the grayscale image
    img_gray_rotated = rotate_image(img_gray, 45)

    # Display the rotated grayscale image
    display_image(img_gray_rotated, "Rotated Grayscale Image")

    # Save the rotated grayscale image
    save_image(img_gray_rotated, "images/lena_gray_rotated.jpg")

    # Show the images
    plt.show() 