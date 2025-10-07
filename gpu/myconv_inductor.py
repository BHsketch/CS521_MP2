import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel
import time

if __name__ == "__main__":
    torch.manual_seed(0)

    # Instantiate your PyTorch model
    N, C, H, W = 1, 2, 8, 8
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=2, out_channels=3, kernel_size=2, stride=1, padding=1).cuda().eval()

    # Torch-Inductor compilation
    '''
    with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=0, active=2, repeat=0),
       record_shapes=False,
       profile_memory=False,
       with_stack=False,
       # on_trace_ready=trace_handler,
   ) as prof_compile:
        with record_function("convolution compile"):
    '''
    start_compile = time.perf_counter()
    scripted_model = torch.compile(model, backend="inductor")
    out = scripted_model(x) # including the first run inside this profile because compilation is lazy
    end_compile = time.perf_counter()

    start_run = time.perf_counter()
    out = scripted_model(x)
    end_run = time.perf_counter()

    compile_time = end_compile - start_compile
    print(f"Compile time: {compile_time:.6f} seconds")

    run_time = (end_run - start_run)
    print(f"Run time: {run_time:.6f} seconds")

    #prof_compile.export_chrome_trace(f"trace_inductor_compile.json")

    '''
    with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=0),
       record_shapes=False,
       profile_memory=False,
       with_stack=False,
       # on_trace_ready=trace_handler,
   ) as prof:
        with record_function("convolution kernel"):
            for step in range(2):
                with record_function(f"iteration_{step}"):
                    out = scripted_model(x)
                    prof.step()

    prof.export_chrome_trace(f"trace_inductor_run.json")
    '''
    
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
