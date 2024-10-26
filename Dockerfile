FROM ubuntu:22.04

WORKDIR /content

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV TERM=xterm-256color
ENV PATH="/home/camenduru/.local/bin:/usr/local/cuda/bin:${PATH}"

RUN apt update -y && apt install -y software-properties-common build-essential \
    libgl1 libglib2.0-0 zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev && \
    add-apt-repository -y ppa:git-core/ppa && apt update -y && \
    apt install -y python-is-python3 python3-pip sudo nano aria2 curl wget git git-lfs unzip unrar ffmpeg && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://developer.download.nvidia.com/compute/cuda/12.6.2/local_installers/cuda_12.6.2_560.35.03_linux.run -d /content -o cuda_12.6.2_560.35.03_linux.run && sh cuda_12.6.2_560.35.03_linux.run --silent --toolkit && \
    echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf && ldconfig && \
    git clone https://github.com/aristocratos/btop /content/btop && cd /content/btop && make && make install && \
    adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home
    
USER camenduru

RUN pip install torch==2.5.0+cu124 torchvision==0.20.0+cu124 torchaudio==2.5.0+cu124 torchtext==0.18.0 torchdata==0.8.0 --extra-index-url https://download.pytorch.org/whl/cu124 && \
    pip install xformers==0.0.28.post2 opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod && \
    pip install diffusers==0.25.0 huggingface_hub==0.24.6 transformers accelerate einops fvcore cloudpickle omegaconf pycocotools scipy onnxruntime scikit-image && \
    git clone https://github.com/yisol/IDM-VTON /content/IDM-VTON && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/image_encoder/config.json -d /content/IDM-VTON/ckpt/vton/image_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/resolve/main/image_encoder/model.safetensors -d /content/IDM-VTON/ckpt/vton/image_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/scheduler/scheduler_config.json -d /content/IDM-VTON/ckpt/vton/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/text_encoder/config.json -d /content/IDM-VTON/ckpt/vton/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/resolve/main/text_encoder/model.safetensors -d /content/IDM-VTON/ckpt/vton/text_encoder -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/text_encoder_2/config.json -d /content/IDM-VTON/ckpt/vton/text_encoder_2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/resolve/main/text_encoder_2/model.safetensors -d /content/IDM-VTON/ckpt/vton/text_encoder_2 -o model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/tokenizer/merges.txt -d /content/IDM-VTON/ckpt/vton/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/tokenizer/special_tokens_map.json -d /content/IDM-VTON/ckpt/vton/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/tokenizer/tokenizer_config.json -d /content/IDM-VTON/ckpt/vton/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/tokenizer/vocab.json -d /content/IDM-VTON/ckpt/vton/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/tokenizer_2/merges.txt -d /content/IDM-VTON/ckpt/vton/tokenizer_2 -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/tokenizer_2/special_tokens_map.json -d /content/IDM-VTON/ckpt/vton/tokenizer_2 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/tokenizer_2/tokenizer_config.json -d /content/IDM-VTON/ckpt/vton/tokenizer_2 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/tokenizer_2/vocab.json -d /content/IDM-VTON/ckpt/vton/tokenizer_2 -o vocab.json && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON/raw/main/unet_dc/config.json -d /content/IDM-VTON/ckpt/vton/unet -o config.json && \
    # aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON/resolve/main/unet_dc/diffusion_pytorch_model.safetensors -d /content/IDM-VTON/ckpt/vton/unet -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/unet/config.json -d /content/IDM-VTON/ckpt/vton/unet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/resolve/main/unet/diffusion_pytorch_model.safetensors -d /content/IDM-VTON/ckpt/vton/unet -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/unet_encoder/config.json -d /content/IDM-VTON/ckpt/vton/unet_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/resolve/main/unet_encoder/diffusion_pytorch_model.safetensors -d /content/IDM-VTON/ckpt/vton/unet_encoder -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/vae/config.json -d /content/IDM-VTON/ckpt/vton/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/resolve/main/vae/diffusion_pytorch_model.safetensors -d /content/IDM-VTON/ckpt/vton/vae -o diffusion_pytorch_model.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON-F16/raw/main/model_index.json -d /content/IDM-VTON/ckpt/vton -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON/resolve/main/densepose/model_final_162be9.pkl -d /content/IDM-VTON/ckpt/densepose -o model_final_162be9.pkl && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON/resolve/main/humanparsing/parsing_atr.onnx -d /content/IDM-VTON/ckpt/humanparsing -o parsing_atr.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON/resolve/main/humanparsing/parsing_lip.onnx -d /content/IDM-VTON/ckpt/humanparsing -o parsing_lip.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/camenduru/IDM-VTON/resolve/main/openpose/ckpts/body_pose_model.pth -d /content/IDM-VTON/ckpt/openpose/ckpts -o body_pose_model.pth

COPY ./worker_runpod.py /content/IDM-VTON/worker_runpod.py
WORKDIR /content/IDM-VTON
CMD python worker_runpod.py