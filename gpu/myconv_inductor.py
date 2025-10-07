import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from myconv import ConvModel

if __name__ == "__main__":
    torch.manual_seed(0)

    # Instantiate your PyTorch model
    N, C, H, W = 2, 3, 19, 19
    x = torch.randn(N, C, H, W).cuda()
    
    model = ConvModel(H, W, in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1).cuda().eval()

    # Torch-Inductor compilation
    scripted_model = torch.compile(model, backend="inductor")

    with torch.profiler.profile(
       activities=[
           torch.profiler.ProfilerActivity.CPU,
           torch.profiler.ProfilerActivity.CUDA,
       ],
       schedule=torch.profiler.schedule(wait=0, warmup=0, active=6, repeat=1),
       record_shapes=True,
       profile_memory=True,
       with_stack=True,
       # on_trace_ready=trace_handler,
   ) as prof:
        with record_function("convolution kernel"):
            out = scripted_model(x)

    prof.export_chrome_trace(f"trace_inductor.json")

    
    # Test your solution
    conv_ref = F.conv2d(x, model.weight, model.bias, stride=1, padding=1)
    print("Inductor --- shape check:", out.shape == conv_ref.shape)
    print("Inductor --- correctness check:", torch.allclose(out, conv_ref, atol=1e-4))
