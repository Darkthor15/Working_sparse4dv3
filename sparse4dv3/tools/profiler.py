import os
import sys
import time
import warnings
import argparse
import pandas as pd
from contextlib import nullcontext
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

import torch
from torch.profiler import profile, ProfilerActivity,  schedule, tensorboard_trace_handler
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv.parallel import scatter
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmcv.runner import wrap_fp16_model

# Set globals
GPU_ID = 0
CONFIG = 'sparse4dv3_temporal_r50_1x8_bs6_256x704'
CKPT = "ckpt/sparse4dv3_r50.pth"
CFG = Config.fromfile(f"projects/configs/{CONFIG}.py")
FP16 = CFG.get("fp16", None)
WAIT = 2
WARMUP = 1
ACTIVE = 1
ACTIVITY = [ProfilerActivity.CPU, ProfilerActivity.CUDA]  
SCHEDULE = schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=1)
TRACER = tensorboard_trace_handler('./log/sparse4dv3')

try:
    from torch2trt import TRTModule
except:
    CFG.use_tensorrt = False

def parse_arguments():
    parser = argparse.ArgumentParser(description="Profile a model")
    parser.add_argument("--profiler", type=int, choices=[0, 1, 2], default=2, help="Profiler type. (0:Simple 1:Pytorch 2:Nsight)")
    return parser.parse_args()

# Build dataloader
def create_dataloader():
    dataset = build_dataset(CFG.data.val)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False,
    )
    return dataloader

# Build data
def create_data():
    dataloader = create_dataloader()
    data_iter = dataloader.__iter__()
    data = next(data_iter)
    data = scatter(data, [GPU_ID])[0]
    return data

# Build model
def create_model():
    model = build_detector(CFG.model, test_cfg=CFG.get("test_cfg"))
    if FP16 is not None:
        wrap_fp16_model(model)
    _ = model.load_state_dict(torch.load(CKPT)["state_dict"], strict=False)
    if CFG.use_tensorrt:
        if CFG.trt_paths["backbone"] is not None:
            model.img_backbone = TRTModule()
            model.img_backbone.load_state_dict(torch.load(CFG.trt_paths["backbone"]))
        if CFG.trt_paths["neck"] is not None:
            model.img_neck = TRTModule()
            model.img_neck.load_state_dict(torch.load(CFG.trt_paths["neck"]))
        if CFG.trt_paths["encoder"] is not None:
            model.head.anchor_encoder = TRTModule()
            model.head.anchor_encoder.load_state_dict(torch.load(CFG.trt_paths["encoder"]))
        if CFG.trt_paths["temp_encoder"] is not None:
            model.head.temp_anchor_encoder = TRTModule()
            model.head.temp_anchor_encoder.load_state_dict(torch.load(CFG.trt_paths["temp_encoder"]))
    model = model.cuda(GPU_ID).eval()
    assert model.use_deformable_func, "Please compile deformable aggregation first !!!"
    return model

def profile_simple():

    loader = create_dataloader()
    model = create_model()

    print(f'{len(loader)} data found!')
    rows = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = scatter(data, [GPU_ID])[0]
            out, profile = model.simple_test_with_profile(return_loss=False, rescale=True, **data)
            profile = { k:round(v,2) for k,v in profile.items() }
            rows.append( (i,) + tuple(v for v in profile.values()) )
            print(f'frame: {i:03d} profile: {profile}')
            sys.stdout.flush()
    df = pd.DataFrame(rows, columns=['FRAME', 'FEAT', 'HEAD', 'POST', 'TOTAL'])
    df.to_csv('inference_time.csv', index=None)

def profile_pytorch():

    data = create_data()
    model = create_model()

    with torch.no_grad():
        with profile(activities=ACTIVITY, schedule=SCHEDULE, on_trace_ready=TRACER, record_shapes=True) as prof:
            for step in range(WAIT+WARMUP+ACTIVE):
                if step < WAIT:
                    print(f"WAIT {step+1}")
                elif step >= WAIT and step < WAIT+WARMUP:
                    print(f"WARMUP {step-WAIT+1}")
                else:
                    print(f"ACTIVE {step-(WAIT+WARMUP)+1}")
                prof.step()
                out = model(return_loss=False, rescale=True, **data)
    print(prof.key_averages().table(sort_by="cuda_time_total"))
    with open('result_sparse4dv3_fp16.txt', 'w') as txt_file:
        txt_file.write(prof.key_averages().table())

def profile_nsys(start_step=1):

    data = create_data()
    model = create_model()

    with torch.no_grad(), torch.cuda.profiler.profile():
        for step in range(start_step+1):
            emit_nvtx = torch.autograd.profiler.emit_nvtx() if step >= start_step else nullcontext()
            with emit_nvtx:
                out = model(return_loss=False, rescale=True, **data)

def profiler_elapsed_time(data, model):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        out = model(return_loss=False, rescale=True, **data)
    torch.cuda.synchronize()
    elapsed_time = time.time() - start
    print(f"Elapsed time: {elapsed_time:.3f} s")

def main():

    args = parse_arguments()
    print('Profiler type:', { 0:'Simple', 1:'Pytorch', 2:'Nsight' }[args.profiler])

    if args.profiler == 0:
        profile_simple()
    elif args.profiler == 1:
        profile_pytorch()
    elif args.profiler == 2:
        profile_nsys()

if __name__ == "__main__":
    main()
