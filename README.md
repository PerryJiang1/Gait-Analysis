
<p align="center">
  <img src="../assets/sapiens_lite_logo.png" alt="Sapiens-Lite" title="Sapiens-Lite" width="500"/>
</p>

## ⚡ Introduction
Sapiens-Lite is our optimized "inference-only" solution, offering:

- Up to 4x faster inference
- Minimal dependencies
- Negligible accuracy loss

## 🚀 Getting Started

- Set the sapiens_lite code root.
  ```bash
  export SAPIENS_LITE_ROOT=$SAPIENS_ROOT/lite
  ```

- We support lite-inference for multiple GPU architectures, primarily in two modes.
  - `MODE=torchscript`: All GPUs with PyTorch2.2+. Inference at `float32`, slower but closest to original model performance.
  - `MODE=bfloat16`: Optimized mode for A100 GPUs with PyTorch-2.3. Uses [FlashAttention](https://github.com/Dao-AILab/flash-attention) for accelerated inference. Coming Soon!

- Note to Windows users: Please use the python scripts in `./demo` instead of `./scripts`.

- Please download the checkpoints from [hugging-face](https://huggingface.co/facebook/sapiens).\
  Checkpoints are suffixed with "_$MODE.pt2".\
  You can be selective about only downloading the checkpoints of interest.\
  Set `$SAPIENS_LITE_CHECKPOINT_ROOT` to the path of `sapiens_lite_host/$MODE`. Checkpoint directory structure:
  ```plaintext
  sapiens_lite_host/
  ├── torchscript
      ├── pretrain/
      │   └── checkpoints/
      │       ├── sapiens_0.3b/
      │       ├── sapiens_0.6b/
      │       ├── sapiens_1b/
      │       └── sapiens_2b/
      ├── pose/
      └── seg/
      └── depth/
      └── normal/
  ├── bfloat16
      ├── pretrain/
      ├── pose/
      └── seg/
      └── depth/
      └── normal/
  ```

## 🔧 Installation
Set up the minimal `sapiens_lite` conda environment (pytorch >= 2.2):
```
conda create -n sapiens_lite python=3.10
conda activate sapiens_lite
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install opencv-python tqdm json-tricks
```

## 🌟 Sapiens-Lite Inference

Note: For inference in `bfloat16` mode:
- Outputs may result in slight variations from the original `float32` predictions.
- The first model run will `autotune` the model and print the log. Subsequent runs automatically load the tuned model.
- Due to `torch.compile` warmup iterations, you'll observe better speedups with a larger number of images, thanks to amortization.

Available tasks:
- ###  [Image Encoder](docs/PRETRAIN_README.md)
- ### [Pose Estimation](docs/POSE_README.md)
- ### [Body Part Segmentation](docs/SEG_README.md)
- ### [Depth Estimation](docs/DEPTH_README.md)
- ### [Surface Normal Estimation](docs/NORMAL_README.md)


## ⚙️ Converting Models to Lite

Obtain a `torch.ExportedProgram` or `torchscript` from the existing sapiens model checkpoint. Note, this requires the full-install `sapiens` conda env.
```bash
cd $SAPIENS_ROOT/scripts/[pretrain,pose,seg]/optimize/local
./[feature_extracter,keypoints*,seg,depth,normal]_optimizer.sh
```
For inference:
- Use `demo.AdhocImageDataset` wrapped with a `DataLoader` for image fetching and preprocessing.\
- Utilize the `WorkerPool` class for multiprocessing capabilities in tasks like saving predictions and visualizations.

## Folder Structure Example

project_root/
├── data/
│   ├── videos/
│   │   └── <VIDEO_NAME>.mp4
│   └── <FRAME_FOLDER>/
│       ├── frame_000001.jpg
│       ├── frame_000002.jpg
│       └── ...
├── output/
│   └── pose/
│       └── <OUTPUT_FOLDER>/
├── demo/
│   ├── vis_pose.py
│   ├── tracker.py
    ├── slider.py
    ├── gait_metrics.py
    └── ...
└── sapiens_checkpoints/
    └── torchscript/
        └── pose/
            └── checkpoints/
                └── sapiens_1b/


## 🧍 Gait Analysis Pipeline

This repository also includes a gait analysis pipeline built on top of Sapiens-Lite pose estimation. The pipeline extracts frames from walking videos, runs Sapiens pose estimation, tracks the selected subject, and computes gait-related metrics such as ankle trajectory, stride timing, and left-right gait symmetry.



### 1. Extract Frames from Video

Use FFmpeg to convert an input walking video into image frames:

```bash
ffmpeg -i "./data/videos/<VIDEO_NAME>.mp4" -q:v 2 "./data/<FRAME_FOLDER>/frame_%06d.jpg"
```


### 2. Extract Skeletons from Frames
```bash
python ./demo/vis_pose.py "<SAPIENS_CHECKPOINT_PATH>" \
  --input "./data/<FRAME_FOLDER>" \
  --output-root "./output/pose/<OUTPUT_FOLDER>" \
  --num_keypoints 17 \
  --shape 1024 768 \
  --device cuda:0 \
  --kpt-thr 0.3 \
  --yolo-model "yolov8m.pt" \
  --batch_size 1
```

### 3. Run Subject Tracking
```bash
python ./demo/tracker.py
```


### 2.5 (optional) Manual Stride Annotation
```bash
python ./demo/slider.py
```

### 3. Compute Gait Metrics

Run the gait analysis script on a tracked pose sequence:

```bash
python ./demo/gait_metrics.py --preset <PRESET_NAME>
```

This will load the predefined configuration for the selected video, including:


track_json_path
metrics_dir
fps
start_frame
end_frame
gt_json_path if available

If gt_json_path is provided, the script will generate the front/back stride analysis with ground-truth vertical markers. Otherwise, it will run the same front/back analysis without ground truth.

To additionally generate the loop-motion video, use:

```bash
python ./demo/gait_metrics.py --preset Baseline_side --run_loop_motion
```

You can also override preset values manually. For example:

```bash
python ./demo/gait_metrics.py --preset Baseline_side --end_frame 150
```

Or run the script without a preset by specifying paths directly:

```bash
python ./demo/gait_metrics.py \
  --track_json_path "./output/pose/<VIDEO_NAME>/tracked/track.json" \
  --metrics_dir "./output/pose/<VIDEO_NAME>/metrics" \
  --fps 30 \
  --start_frame 20 \
  --end_frame 115
```

With loop-motion video:

```bash
python .\demo\gait_metrics.py --preset Baseline_side --run_loop_motion
```

The script generates gait-related plots and summary files in the metrics output folder, including:

17_keypoints_time_vs_y.png
front/back stride plots
step timing and height visualizations
wrist_relative_height.png
metrics.txt
optional loop-motion video
