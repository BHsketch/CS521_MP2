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
    num_pixels_per_in_channel = input_height*input_width
    img_padding = ((filter_height -1)*input_width + filter_width - 1)
    padded_img_tile_row = num_pixels_per_in_channel + img_padding
    padded_img_tile_size = tile_size_pixels + img_padding
    shift_ij = img_padding
    elements_per_filter = filter_height*filter_width

    tile_size_pixels = nl.tile_size.gemm_moving_fmax - img_padding
    X_re = X.reshape((batch_size, in_channels, (input_height*input_width)))         # all pixels will be aranged in just one dimension
    W_re = W.reshape((out_channels, in_channels, (filter_height*filter_width)))     
    X_out_re = X_out.reshape((batch_size, out_channels, out_pool_height*out_pool_width)) 

    # Note: We are loading the image entire input channels at a time, but multiplying them 128x512 elements at a time
    # Idea is to reduce the total number of DMA accesses. This is the reason the load is not done in the same loop
    # body as the iteration
    
    # output = nl.zeros((batch_size, out_channels, out_pool_height, out_pool_width),dtype=X_out.dtype, buffer=hbm)

    # out_channels
    # pixels
    # in channels
    # f_hw

    weights_tile = nl.ndarray((c_in_pmax, elements_per_filter), dtype=W_re.dtype, buffer=nl.sbuf)
    image_tile = nl.ndarray((c_in_pmax, padded_img_tile_size), dtype=X_re.dtype, buffer=nl.sbuf)

    # Process the images in batches
    for b in nl.affine_range(batch_size):
        # raise RuntimeError("Please fill your implementation of computing convolution"
                           # " of X[b] with the weights W and bias b and store the result in X_out[b]")

        for o in nl.sequential_range(out_channels): # just one pixel depth at a time
            
            for p in nl.sequential_range(num_pixels_per_in_channel // tile_size_pixels):

                # TODO Allocate output matrix
                res_psum = nl.zeros((1, tile_size_pixels), nl.float32, buffer=nl.psum)

                for i in nl.sequential_range(in_channels // c_in_pmax):

                    # TODO fetch tile from both weight matrix and image matrix

                    weights_tile[...] = nl.load(W_re[o, (c_in_pmax*i):(c_in_pmax*(i+1)), :])
                    
                    if(p == ((num_pixels_per_in_channel // tile_size_pixels)-1)):
                        image_tile[:, 0:padded_img_tile_size] = nl.load(X_re[b, (c_in_pmax*i):(c_in_pmax*(i+1)), (tile_size_pixels*p):(tile_size_pixels*(p+1) + img_padding)])   
                    else:
                        image_tile[:, 0:tile_size_pixels] = nl.load(X_re[b, (c_in_pmax*i):(c_in_pmax*(i+1)), (tile_size_pixels*p):(tile_size_pixels*(p+1))])   
                        image_tile[:, tile_size_pixels:padded_img_tile_size] = nl.zeros((c_in_pmax, img_padding), image_tile.dtype, buffer=nl.sbuf)
                    
                    for filter_i in nl.sequential_range(filter_height):
                        for filter_j in nl.sequential_range(filter_width):

                            shift_ij = (filter_i*input_width + filter_j)
                            filter_pixel = (filter_i*filter_width) + filter_j

                            res_psum += nl.matmul( weights_tile[:, filter_pixel], image_tile[:, (shift_ij):(tile_size_pixels + shift_ij)] , transpose_x = True)
                    
                
                res_sb = nl.copy(res_psum, dtype=X_out.dtype)
                nl.store(X_out[b, o, (tile_size_pixels*p):(tile_size_pixels*(p+1))],
               value=res_sb)
    
    X_out = X_out_re.reshape((batch_size, out_channels, out_pool_height, out_pool_width))
        # -------------- OLD CODE ---------------------

        # for filter_i in nl.affine_range(filter_height): # TODO Fill this in

            # for filter_j in nl.affine_range(filter_width): # TODO Fill this in

                # #TODO Shifting logic for th image matrix:
                # shift_ij = ((filter_i -1)*input_width + filter_j - 1)

                # #TODO Allocate a 128xnum_pixels_per_in_channel sized matrix in SBUF, and a 128x128 weight matrix too
                # weights_tile = nl.ndarray((c_out_pmax, c_in_pmax), dtype=W.dtype, buffer=nl.sbuf)

                # # padding the image tile with zeroes on the right to allow for shifting
                # image_tile = nl.ndarray((c_in_pmax, padded_img_tile_row), dtype=X_re.dtype, buffer=nl.sbuf) 
                
                # # process 128 input channels at a time. This will be the partition dimension
                # for i in nl.affine_range(in_channels // c_in_pmax):

                    # # TODO Bring in 128 ENTIRE rows of the image matrix into SBUF
                    # image_tile[:, 0:num_pixels_per_in_channel] = nl.load(X_re[b, (c_in_pmax*i):(c_in_pmax*(i+1)), :])   # bring in only the appropriate 128-element tile 
                                                                                                                        # # in the in_channels dimension
                                                                                                                        # # but bring in everything from the pixels dimensions

                    # image_tile[:, num_pixels_per_in_channel:padded_img_tile_row] = nl.zeros((c_in_pmax, img_padding), image_tile.dtype, buffer=nl.sbuf)

                    # # Reason: contiguous memory, plus taking advantage of the fact that the free dimension can be almost arbitrarily long.
                    # for o in nl.affine_range(out_channels // c_out_pmax):
                        
                        # # TODO Bring in a 128x128 grid of the weights matrix into SBUF, corresponding to in and out channels
                        # weights_tile = nl.load(W_re[(c_out_pmax*o):(c_out_pmax*(o+1)), (c_in_pmax*i):(c_in_pmax*(i+1)), (filter_i*filter_width + filter_j)])
                        

                        # # In the free dimension, we can tile 512 at a time.
                        # for p in nl.affine_range(num_pixels_per_in_channel // tile_size_pixels):
                            # res_psum = nl.zeros((c_in_pmax, tile_size_pixels), nl.float32, buffer=nl.psum) 
                            # res_psum += nl.matmul(weights_tile[...], image_tile[:, (p*(tile_size_pixels) + shift_ij):((p+1)*tile_size_pixels+shift_ij)], transpose_x=False)
        # --------------------------------------------

    return X_out

