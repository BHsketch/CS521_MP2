import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
import math

device = torch.device("cuda:0")
# device = torch.device("cpu")

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
        self.out_h = (H + 2*padding - kernel_size) // stride + 1
        self.out_w = (W + 2*padding - kernel_size) // stride + 1

        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, kernel_size, kernel_size), device=device))
        self.bias = nn.Parameter(torch.zeros((out_channels), device=device))

        

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
        patches = torch.zeros((N, out_h*out_w, C*KH*KW), device=device)

        # iterate over all submatrices to be extracted
        for n in range(0, N):
            for output_i in range(0, out_h):
                for output_j in range(0, out_w):
                    
                    input_start_i = output_i*S
                    input_start_j = output_j*S

                    # X shape is (N, C, H, W)
                    # extracting a chunk that when multiplied with the kernel, results in a single output element
                    input_subtensor = x_pad[n, :, (input_start_i):(input_start_i+KH), (input_start_j):(input_start_j+KW)]

                    # convert 3d subtensor into a single row
                    input_subtensor_flattened = input_subtensor.reshape(-1) 

                    # Each block from the input should form one column in the output
                    patches[n, (output_i*out_w + output_j),:] = input_subtensor_flattened


        # print("patches: \n", patches, "\n")
        return patches

    def conv2d_manual(self, x):
        N = x.shape[0]
        C_out = self.out_channels
        KH = KW = self.kernel_size

        # TO DO: 1) convert input (x) into shape (N, out_h*out_w, C*KH*KW).
        cols = self.im2col_manual(x)
        cols_t = torch.transpose(cols, 1, 2)
        # print("cols_t: \n", cols_t, "\n")

        # TO DO: 2) flatten self.weight into shape (C_out, C*KH*KW).
        weights_flattened = self.weight.reshape((C_out, (self.in_channels)*KH*KW))
        # print("weights_flattened:\n", weights_flattened, "\n")

        # TO DO: 3) perform tiled matmul after required reshaping is done.
        tile_size_i = 15
        tile_size_j = 15
        tile_size_k = 15
        max_i = C_out 
        max_j = (self.out_h*self.out_w)
        max_k = (self.in_channels*self.kernel_size*self.kernel_size)
        num_i_tiles = math.ceil(max_i / tile_size_i)
        num_j_tiles = math.ceil(max_j / tile_size_j)
        num_k_tiles = math.ceil(max_k / tile_size_k)
        
        output = torch.zeros((N, C_out,(self.out_h*self.out_w)), device=device)

        for ii in range(num_i_tiles):
            for jj in range(num_j_tiles):
                for kk in range(num_k_tiles):
                    limit_i = min(tile_size_i*(ii+1), max_i)
                    limit_j = min(tile_size_j*(jj+1), max_j)
                    limit_k = min(tile_size_k*(kk+1), max_k)
                    # if(ii == 0 and jj == 0):
                        # print("first weights tile: ", weights_flattened[(tile_size_i*ii):(limit_i), :])
                        # print("first output tile: ", cols_t[:,:, (tile_size_j*jj):(limit_j)])
                    output[:, (tile_size_i*ii):(limit_i), (tile_size_j*jj):(limit_j)] += torch.matmul(weights_flattened[(tile_size_i*ii):(limit_i), (tile_size_k*kk):(limit_k)], cols_t[:,(tile_size_k*kk):(limit_k), (tile_size_j*jj):(limit_j)])
                    # output[n, :,:] = torch.matmul(weights_flattened[(tile_size_i*ii):(limit_i), (tile_size_k*kk):(limit_k)], cols[n,(tile_size_k*kk):(limit_k), (tile_size_j*jj):(limit_j)])
                    # for i in range(limit_i):
                        # for j in range(limit_j):
                            # for k in range(limit_k):
                                # output[n, i, j] += weight_flattened[i, k] * cols[n, k, j]


        # TO DO: 4) Add bias.
        bias_col = self.bias.reshape((C_out, 1))
        final_out = output + bias_col

        # TO DO: 5) reshape output into shape (N, C_out, out_h, out_w).
        final_out = final_out.reshape((N, C_out, self.out_h, self.out_w)) 

        # print("final output: \n", final_out, "\n")
        return final_out
        #return out

    def forward(self, x):
        return self.conv2d_manual(x)

# def trace_handler(prof: torch.profiler.profile):
   # # Prefix for file names.
   # host_name = socket.gethostname()
   # timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   # file_prefix = f"{host_name}_{timestamp}"

   # # Construct the trace file.
   # prof.export_chrome_trace(f"{file_prefix}.json.gz")

   # # Construct the memory timeline file.
   # prof.export_memory_timeline(f"{file_prefix}.html", device="cpu")
                               # #device="cuda:0")

if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, H, W = 3, 4, 32, 32
    x = torch.randn(N, C, H, W, device=device) 
    # print("X: \n", x, "\n")
    out_channels=8
    kernel_size=8
    model = ConvModel(H, W, C, out_channels, kernel_size, stride=1, padding=1).to(device)

    # ----------
    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        # with record_function("model_inference"):

    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    # prof.export_chrome_trace("trace.json")
    # ----------

    with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=2, active=6, repeat=1),
       record_shapes=True,
       profile_memory=True,
       with_stack=True,
       # on_trace_ready=trace_handler,
   ) as prof:
        with record_function("convolution kernel"):
            for step in range(10):
                out = model(x)
                prof.step()

    prof.export_chrome_trace(f"trace_interpreter.json")


    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    # print("reference output: \n", conv_ref)
    print("output:\n",out,"\n")
    print("reference:\n",conv_ref, "\n")
    print("PyTorch --- shape check:", out.shape == conv_ref.shape)
    print("PyTorch --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
