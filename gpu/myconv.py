import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda:0")

class ConvModel(nn.Module):
    def __init__(self, H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.stride = stride
        self.padding = padding

        self.H = H
        self.W = W

        # TO DO: Define static shapes here. 

        # Precompute output size
        self.out_h = (H - kernel_size + 1) // stride
        self.out_w = (W - kernel_size + 1) // stride

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        

    def im2col_manual(self, x):
        N = x.shape[0]        # batch size can remain dynamic
        C = self.in_channels
        KH = KW = self.kernel_size
        S = self.stride
        P = self.padding
        out_h = self.out_h
        out_w = self.out_w

        # Pad input
        x_pad = F.pad(x, (P, P, P, P))

        # TO DO: Convert input (x) into shape (N, out_h*out_w, C*KH*KW). 
        # Refer to Lecture 3 for implementing this operation.
        patches = torch.zeros((N, out_h*out_w, C*KH*KW))

        # iterate over all submatrices to be extracted
        for n in range(0, N):
            for output_i in range(0, out_h):
                for output_j in range(0, out_w):
                    
                    input_start_i = output_i*S
                    input_start_j = output_j*S

                    # X shape is (N, C, H, W)
                    # extracting a chunk that when multiplied with the kernel, results in a single output element
                    input_subtensor = x[n, :, (input_start_i):(input_start_i+KH), (input_start_j):(input_start_j+KW)]

                    # convert 3d subtensor into a single row
                    input_subtensor_flattened = input_subtensor.reshape((C*KH*KW, 1))

                    # Each block from the input should form one column in the output
                    patches[n, (output_i*out_w + output_j),:] = input_subtensor_flattened.permute(1, 0) 


        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        KH = KW = self.kernel_size

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)          

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        weight_flattened = self.weight.reshape((C_out, C*KH*KW))

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        tile_size_i = 8
        tile_size_j = 8
        tile_size_k = 8
        max_i = C_out 
        max_j = (self.out_h*self.out_w)
        max_k = (self.in_channels*self.kernel_size*self.kernel_size)
        num_i_tiles = max_i // tile_size_i
        num_j_tiles = max_j // tile_size_j
        num_k_tiles = max_k // tile_size_k
        
        output = torch.zeros((N, C_out,(self.out_h*self.out_w)))
        for n in range(N):
            for ii in range(num_i_tiles):
                for jj in range(num_j_tiles):
                    for kk in range(num_k_tiles):
                        limit_i = min(tile_size_i*(ii+1), max_i)
                        limit_j = min(tile_size_j*(jj+1), max_j)
                        limit_k = min(tile_size_k*(kk+1), max_k)
                        for i in range(limit_i):
                            for j in range(limit_j):
                                for k in range(limit_k):
                                    output[n, i, j] += weight_flattened[i, k] * cols[n, k, j]


        # TO DO: 4) Add bias.
        bias_col = self.bias.reshape((out_channels, 1))
        final_out = output + bias_col

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        final_out = final_out.reshape((N, C_out, out_h, out_w)) 

        return final_out
        #return out

    def forward(self, x):
        return self.conv2d_manual(x)

if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 2, 4, 22, 22
    x = torch.randn(N, C, H, W)
    out_channels=8
    kernel_size=7
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1)
    out = model(x)

    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
