import numpy as np
import math

import neuronxcc.nki as nki
import neuronxcc.nki.language as nl
import neuronxcc.nki.isa as nisa
from neuronxcc.nki import baremetal


"""
A convolution kernel that you need to implement.

Parameters:
    X: the input tensor
    W: the weights of the convolution filters.
    bias: the biases of the convolution filters.

expect: X.shape == [batch_size, in_channels, input_height, input_width]
expect: W.shape == [out_channels, in_channels, filter_height, filter_width]
expect: bias.shape == [out_channels]
expect: filter_height == filter_width
expect: input_channels % 128 == 0
expect: output_channels % 128 == 0

out_height = input_height - filter_height + 1
out_width = input_width - filter_width + 1

out_pool_height = out_height
out_pool_width = out_width

The shape of the output should be [batch_size, out_channels, out_pool_height, out_pool_width]


### ALGORITHM ###

for b in batches:
    for o in out_channels:
        - bring in the out-channel slice from weight matrix, with out channels being nl.par_dim
        for each slice in the pixels dimension // tile defined as rows 
            - bring in tiles for all input channels, with tile size being rows 
            - allocate the psum matrix
            for filter_i in filter_width:
                for filter_j in filter_width:
                    for i in input_channels:
                        - bring in tile 1
                        - bring in tile 2
                        - psum += nl.matmul(weightSubtensor, shiftedImageTensor)
            - (add bias to the psum matrix)
            - store psum output to hbm directly

#################

"""

@nki.jit
def conv2d(X, W, bias):

    batch_size, in_channels, input_height, input_width = X.shape
    out_channels, in_channels_, filter_height, filter_width = W.shape
    out_channels_ = bias.shape[0]

    assert (
        in_channels_ == in_channels and out_channels_ == out_channels
    ), f"Shape mismatch. {in_channels}, {in_channels_}, {out_channels}, {out_channels_}"

    out_height = input_height - filter_height + 1
    out_width = input_width - filter_width + 1

    out_pool_height = out_height
    out_pool_width = out_width
    
    # Can assume multiple of 128 to avoid using mask
    assert in_channels % 128 == 0

    # Can assume one PSUM bank can at least fit one row of the pixels
    assert nl.tile_size.gemm_moving_fmax >= out_width

    # Initialize output array
    X_out = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height, out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    X_out_re = nl.ndarray(
        shape=(batch_size, out_channels, out_pool_height*out_pool_width),
        dtype=X.dtype,
        buffer=nl.hbm,
    )

    # Reshaping some inputs
    X_re = X.reshape((batch_size, in_channels, (input_height*input_width)))         # all pixels will be aranged in just one dimension
    W_re = W.reshape((out_channels, in_channels, (filter_height*filter_width)))     

    # Constants
    num_pixels_per_in_channel = input_height*input_width
    num_pixels_in_output = out_height*out_width
    num_elements_in_filter = filter_height*filter_width
    img_padding = ((filter_height -1)*input_width + filter_width - 1)

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax 
    c_out_pmax = nl.tile_size.pmax
    # TODO check this to be sure later
    tile_size_pixels = 2*out_width
    padded_tile_size_pixels = tile_size_pixels + img_padding 

    # Shape parameters
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_c_out = out_channels // c_out_pmax
    n_tiles_pixels = num_pixels_in_output // tile_size_pixels   # Note, we tile by output pixels so that 
                                                                # we can stop iterating at the right place wrt
                                                                # the input matrix

    # Allocating weight and img tiles
    weights_slice = nl.ndarray((nl.par_dim(c_out_pmax), in_channels, num_elements_in_filter), dtype=W_re.dtype, buffer=nl.sbuf)
    image_tile = nl.ndarray((nl.par_dim(c_in_pmax), padded_tile_size_pixels), dtype=X_re.dtype, buffer=nl.sbuf)
    
    for b in nl.sequential_range(batch_size):
        # Iterate over output channels
        for o in nl.sequential_range(n_tiles_c_out):
            # bring in the entire subtensor required to compute the first output tile
            weights_slice[:, :, :] = nl.load(W_re[(c_out_pmax*o):(c_out_pmax*(o+1)),:,:])

            for p in nl.sequential_range(n_tiles_pixels):
                # TODO mark this as par_dim?
                res_psum = nl.zeros((c_out_pmax, tile_size_pixels), nl.float32, buffer=nl.psum) 

                for i in nl.sequential_range(n_tiles_c_in):
                    # bring in the necessary pixels: a tile plus some amount corresponding to the shift
                    # I think something is wrong with the tile indexing logic here
                    image_tile[:, :] = nl.load(X_re[b, (c_in_pmax*i):(c_in_pmax*(i+1)), (tile_size_pixels*p):(tile_size_pixels*(p+1) + img_padding)])

                    for filter_i in nl.sequential_range(filter_height):
                        for filter_j in nl.sequential_range(filter_width):
                            shift_ij = (filter_i*input_width + filter_j)
                            res_psum += nl.matmul(weights_slice[:, (c_in_pmax*i):(c_in_pmax*(i+1)), (filter_i*filter_width + filter_j)], image_tile[:, shift_ij:(tile_size_pixels + shift_ij)])
                
                nl.store(X_out_re[b, (c_out_pmax*o):(c_out_pmax*(o+1)), (tile_size_pixels*p):(tile_size_pixels*(p+1))], value=res_psum)

    X_out = X_out_re.reshape((batch_size, out_channels, out_pool_height, out_pool_width))

    return X_out

