import json

import numpy as np
from scipy import signal
from skimage.io import imsave, imshow, show, imread
from matplotlib import pyplot as plt

settings = {
    'filepath': 'D:\\Рабочий стол\\TestImages\\14_LENA.TIF',
    'path_to_save_result': "D:\\Рабочий стол\\Images"
}


MAX_BRIGHTNESS_VALUE = 255
MIN_BRIGHTNESS_VALUE = 0
BORDER_PROCESSING_PARAMETER = 20
WINDOW_HORIZONTAL = np.array([[-1, 1]])
WINDOW_VERTICAL = np.array([[-1], [1]])


def border_processing_function(element_value):
    if element_value < BORDER_PROCESSING_PARAMETER:
        return MIN_BRIGHTNESS_VALUE
    else:
        return MAX_BRIGHTNESS_VALUE


def border_processing(img_as_arrays):
    shape = np.shape(img_as_arrays)
    new_img_list = list(map(border_processing_function, np.reshape(img_as_arrays, img_as_arrays.size)))
    single_dimension_array = np.array(new_img_list)
    new_img = np.reshape(single_dimension_array, (shape[0], shape[1]))
    return new_img


def outline_selection(matrix_u, matrix_v):
    sum_of_squares = np.square(matrix_u) + np.square(matrix_v)
    return np.sqrt(sum_of_squares)


def window_processing(matrix, window):
    return signal.convolve2d(matrix, window, boundary='symm', mode='same')


def create_figure_of_union_plot(src_img, deriv_horiz, grad_matrix, deriv_vert, img_border):
    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(2, 3, 1)
    plt.title("Source image")
    imshow(src_img, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 2)
    plt.title("Derivative horizontal")
    imshow(deriv_horiz, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 3)
    plt.title("Gradient evaluation")
    imshow(grad_matrix, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 4)
    plt.title("Border processing")
    imshow(img_border, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 5)
    plt.title("Derivative vertical")
    imshow(deriv_vert, cmap='gray', vmin=0, vmax=255)
    fig.add_subplot(2, 3, 6)
    plt.title("Histogram of gradient evaluation")
    plt.xlabel('Brightness values')
    plt.ylabel('Pixels quantity')
    create_wb_histogram_plot(grad_matrix)
    return fig


def create_wb_histogram_plot(img_as_arrays):
    hist, bins = np.histogram(img_as_arrays.flatten(), 256, [0, 256])
    plt.plot(bins[:-1], hist, color='blue', linestyle='-', linewidth=1)


def open_image_as_arrays(filepath):
    return imread(filepath)


def save_image(img, directory):
    imsave(directory, img)


def save_current_parameters_to_file(filepath):
    json.dump(settings, open(filepath, 'w'))


save_current_parameters_to_file('D:\\Рабочий стол\\Lab_2_settings.json')
json_settings_file = json.load(open('D:\\Рабочий стол\\Lab_2_settings.json'))
image_filepath = json_settings_file['filepath']
path_to_save = json_settings_file['path_to_save_result']


img_as_array = open_image_as_arrays(image_filepath)

a = np.array([[0, 1, 2, 4, 7, 10, 15], [0, 1, 2, 4, 7, 10, 15], [0, 1, 2, 4, 7, 10, 15], [80, 70, 60, 50, 40, 30, 20], [80, 70, 60, 50, 40, 30, 20]])


print(a)
grad = signal.convolve2d(a, WINDOW_HORIZONTAL, boundary='symm', mode='same')
print("Convolve result")
print(grad)
print("Another")
print(a)
print("Convolve result")

grad = signal.convolve2d(a, WINDOW_VERTICAL, boundary='symm', mode='same')
print(grad)
print(np.abs(grad))
print("New age begins")

img_derivative_horizontal = window_processing(img_as_array, WINDOW_HORIZONTAL)
img_derivative_vertical = window_processing(img_as_array, WINDOW_VERTICAL)
img_gradient = outline_selection(img_derivative_horizontal, img_derivative_vertical)

create_figure_of_union_plot(img_as_array, np.abs(img_derivative_horizontal), img_gradient, np.abs(img_derivative_vertical), border_processing(img_gradient))
plt.tight_layout()
show()
