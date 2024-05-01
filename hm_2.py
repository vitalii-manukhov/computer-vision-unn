import numpy as np

def Convolution(input_tensor, filters):
    height_of_input = input_tensor.shape[0]
    width_of_input = input_tensor.shape[1]
    number_of_channels = input_tensor.shape[2]
    height_of_filter = filters[0].shape[0]
    width_of_filter = filters[0].shape[1]
    number_of_filters = filters.shape[0]

    output = np.zeros((height_of_input - height_of_filter + 1,
                       width_of_input - width_of_filter + 1,
                       number_of_filters))

    for filter_index in range(number_of_filters):
        for input_height in range(height_of_input - height_of_filter + 1):
            for input_width in range(width_of_input - width_of_filter + 1):
                for input_channel in range(number_of_channels):
                    for filter_height in range(height_of_filter):
                        for filter_width in range(width_of_filter):
                            output[input_height, input_width, filter_index] +=\
                                input_tensor[input_height + filter_height, input_width + filter_width, input_channel] *\
                                    filters[filter_index, filter_height, filter_width, input_channel]

    return output

def ConvolutionWithIm2Col(input_tensor, filters):
    height_of_input = input_tensor.shape[0]
    weight_of_input = input_tensor.shape[1]
    number_of_channels = input_tensor.shape[2]
    height_of_filter = filters[0].shape[0]
    weight_of_filter = filters[0].shape[1]
    number_of_filters = filters.shape[0]

    left_matrix = filters.reshape((number_of_filters, -1))

    patches = []
    for i in range(0, height_of_input - height_of_filter + 1):
        for j in range(0, weight_of_input - weight_of_filter + 1):
            temporary_column = []
            for c in range(0, number_of_channels):
                patch = input_tensor[i:i + height_of_filter, j:j + weight_of_filter, c]
                flattened_patch = patch.flatten()
                temporary_column.extend(flattened_patch)

            patches.append(temporary_column)

    patches = np.array(patches)
    output = patches @ left_matrix.T

    return output.reshape(height_of_input - height_of_filter + 1, weight_of_input - weight_of_filter + 1, number_of_filters)

np.random.seed(42)
image = np.random.uniform(low=0, high=255, size=(3, 3, 1)) 
filters = np.random.uniform(low=0, high=255, size=(1, 3, 3, 1)) 

result_1 = Convolution(image, filters)
print(result_1)
result_2 = ConvolutionWithIm2Col(image, filters)
print(result_2)