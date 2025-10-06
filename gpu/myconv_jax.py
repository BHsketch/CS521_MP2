import jax
import jax.numpy as jnp
from jax import jit
import torch.nn.functional as F
import numpy as np
import torch
from myconv import ConvModel
import jax.profiler
from jax import lax

# Create a log directory
logdir = "./jax_trace"

def im2col_manual_jax(x, KH, KW, S, P, out_h, out_w):
    ''' 
        Reimplement the same function (im2col_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
    '''
    # x: (N, C, H, W)
    N, C, H, W = x.shape
    
    # Pad input
    x_pad = jnp.pad(x, ((0,0),(0,0),(P,P),(P,P)))

    # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
    # Refer to Lecture 3 for implementing this operation.
    patches = jnp.zeros((N, out_h*out_w, C*KH*KW))

    def update_cols_batch(b, colsBatch):
        def update_cols_out_ele(ele, block):
            input_start_i = (ele // out_w)*S
            input_start_j = (ele - (ele // out_w)*out_w)*S

            patch = lax.dynamic_slice(x_pad, (b, 0, input_start_i, input_start_j), (1, C, KH, KW)).reshape(-1)

            return block.at[b, ele, :].set(patch)

            # return block.at[b, ele, :].set(x_pad[b, :, (input_start_i):(input_start_i+KH), (input_start_j):(input_start_j+KW)].reshape(-1))

        return jax.lax.fori_loop(0, out_h*out_w, update_cols_out_ele, colsBatch)
    
    patches = jax.lax.fori_loop(0, N, update_cols_batch, patches)

    # patches = ...
    return patches

def conv2d_manual_jax(x, weight, bias, stride=1, padding=1):
    '''
        Reimplement the same function (conv2d_manual) in myconv.py "for JAX". 
        Hint: Instead of torch tensors, use of jnp arrays is required to leverage JIT compilation and GPU execution in JAX
        Hint: Unlike PyTorch, JAX arrays are immutable, so you cannot do indexing like out[i:j, :] = ... inside a JIT. You may use .at[].set() instead.
    '''
    N, C, H, W = x.shape
    C_out, _, KH, KW = weight.shape

    # define your helper variables here
    out_h = (H + 2*padding - KH) // stride + 1

    out_w = (W + 2*padding - KW) // stride + 1
    
    # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
    cols = im2col_manual_jax(x, KH, KW, stride, padding, out_h, out_w)

    # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
    weights_flattened = weight.reshape((C_out, C*KH*KW))

    # TO DO: 3) perform tiled matmul after required reshaping is done.

    def matmul_tiled_batch(b, resultForBatch): 
        def matmul_tiled(output_channel, resultForOC): 
            return resultForOC.at[b, output_channel, :].set(jnp.add(jnp.matmul(cols[b, :, :], weights_flattened[output_channel, :]), bias[output_channel])) 
        return jax.lax.fori_loop(0, C_out, matmul_tiled, resultForBatch)

    result = jnp.zeros((N, C_out, out_h*out_w))
    result = jax.lax.fori_loop(0, N, matmul_tiled_batch, result)

    # TO DO: 4) Add bias.// done in the above function

    # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
    out = result.reshape((N, C_out, out_h, out_w))


    return out

if __name__ == "__main__":
    # Instantiate PyTorch model
    H, W = 33, 33
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=5, stride=1, padding=1)
    model.eval()

    # Example input
    x_torch = torch.randn(1, 3, H, W)

    # Export weights and biases
    params = {
        "weight": model.weight.detach().cpu().numpy(),  # shape (out_channels, in_channels, KH, KW)
        "bias": model.bias.detach().cpu().numpy()       # shape (out_channels,)
    }

    # Convert model input, weights and bias into jax arrays
    x_jax = jnp.array(x_torch.numpy())
    weight_jax = jnp.array(params["weight"])
    bias_jax = jnp.array(params["bias"])

    # enable JIT compilation
    conv2d_manual_jax_jit = jit(conv2d_manual_jax)

    # call your JAX function
    out_jax = conv2d_manual_jax_jit(x_jax, weight_jax, bias_jax)
    out_np = jax.device_get(out_jax)
    out_torch = torch.from_numpy(out_np)
    # Test your solution
    conv_ref = F.conv2d(x_torch, model.weight, model.bias, stride=1, padding=1)
    # conv_ref_jax = jnp.array(conv_ref.numpy())
    print("JAX --- shape check:", out_torch.shape == conv_ref.shape)
    print("JAX --- correctness check:", torch.allclose(out_torch, conv_ref, atol=1e-1))
