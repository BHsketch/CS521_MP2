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

    # Various tiling dimensions (You may want to define more of them)
    c_in_pmax = nl.tile_size.pmax
    c_out_pmax = nl.tile_size.pmax
    n_tiles_c_in = in_channels // c_in_pmax
    n_tiles_pixels = nl.tile_size.gemm_moving_fmax
    num_pixels_per_in_channel = input_height*input_width

    X_re = X.reshape((batch_size, in_channels, (input_height*input_width)))         # all pixels will be aranged in just one dimension
    W_re = W.reshape((out_channels, in_channels, (filter_height*filter_width)))     

    # Note: We are loading the image entire input channels at a time, but multiplying them 128x512 elements at a time
    # Idea is to reduce the total number of DMA accesses. This is the reason the load is not done in the same loop
    # body as the iteration

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # raise RuntimeError("Please fill your implementation of computing convolution"
                           # " of X[b] with the weights W and bias b and store the result in X_out[b]")
        for filter_i in nl.affine_range(): # TODO Fill this in

            for filter_j in nl.affine_range(): # TODO Fill this in

                #TODO Shifting logic for th image matrix

                #TODO Allocate a 128xnum_pixels_per_in_channel sized matrix in SBUF, and a 128x128 weight matrix too
                weights_tile = nl.ndarray((c_out_pmax, c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)
                image_tile = nl.ndarray((c_in_pmax, num_pixels_per_in_channel), dtype=X_re.dtype, buffer=nl.sbuf)
                
                # process 128 input channels at a time. This will be the partition dimension
                for i in nl.affine_range(in_channels // n_tiles_c_in):

                    # TODO Bring in 128 ENTIRE rows of the image matrix into SBUF
                    image_tile = nl.load(X_re[b, (c_in_pmax*i):(c_in_pmax*(i+1)), :])  # bring in only the appropriate 128-element tile 
                                                                                    # in the in_channels dimension
                                                                                    # but bring in everything from the pixels dimensions

                    # Reason: contiguous memory, plus taking advantage of the fact that the free dimension can be almost arbitrarily long.
                    for o in nl.affine_range(out_channels // c_out_pmax):
                        
                        # TODO Bring in a 128x128 grid of the weights matrix into SBUF, corresponding to in and out channels

                        # In the free dimension, we can tile 512 at a time.
                       for p in nl.affine_range(num_pixels_per_in_channel // n_tiles_pixels):


    return X_out

