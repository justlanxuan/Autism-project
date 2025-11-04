# A Multimodal Wearable and Camera-Based System for Behavior Monitoring and Personalized Intervention in Children with Autism

Workflow in linux
1. Replace config settings at config.yaml
other key words: "YOLO path"
2. setup mmpose
```
conda create -n imu python=3.10 -y
conda activate imu

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

pip install openmim
mim install "mmengine>=0.10.0,<1.0.0"
mim install "mmcv>=2.1.0,<2.2.0"
mim install "mmpose>=1.3.0,<2.0.0"
```
3. other setups
```
conda install -r requirements.txt
```
3. try test.py

4. 

``` 
