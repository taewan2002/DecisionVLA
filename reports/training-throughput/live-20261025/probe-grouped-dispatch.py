import os,time,json
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
import torch
torch.cuda.set_per_process_memory_fraction(0.02,0)
torch.use_deterministic_algorithms(True)
a=torch.ones((1024,128),device="cuda",dtype=torch.bfloat16)
b=torch.ones((128,64,128),device="cuda",dtype=torch.bfloat16).transpose(-1,-2)
offs=torch.arange(8,1025,8,device="cuda",dtype=torch.int32)
y=torch._grouped_mm(a,b,offs=offs);torch.cuda.synchronize()
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
 y=torch._grouped_mm(a,b,offs=offs)
 torch.cuda.synchronize()
assert torch.all(y==128).item()
rows=[{"op":x.key,"count":x.count,"cpu_total_us":x.cpu_time_total,"self_cpu_us":x.self_cpu_time_total} for x in p.key_averages()]
print(json.dumps({"torch":torch.__version__,"git_revision":torch.version.git_version,"gpu":torch.cuda.get_device_name(),"capability":torch.cuda.get_device_capability(),"input_shape":list(a.shape),"weight_shape":list(b.shape),"groups":128,"correct_output":True,"scope":"isolated synthetic BF16 grouped-mm forward dispatch probe, no model loading/training; concurrent main run timings excluded","peak_allocated_bytes":torch.cuda.max_memory_allocated(),"operators":rows},indent=2))
