ARG PYTORCH="2.0.0"
ARG CUDA="11.7"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y wget git ninja-build unzip libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV FORCE_CUDA="1"
ENV CUDA_HOME=/usr/local/cuda
RUN pip install fvcore iopath
RUN wget https://github.com/NVIDIA/cub/archive/refs/tags/1.17.2.tar.gz && tar xzf 1.17.2.tar.gz
ENV CUB_HOME=/workspace/cub-1.17.2
ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
RUN pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu117_pyt200/download.html

RUN MMCV_WITH_OPS=1 pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch${PYTORCH}/index.html
RUN pip install mmdet==3.1.0 mmpose==1.3.0 mmengine==0.8.3 mmpretrain==1.2.0

RUN pip install https://data.pyg.org/whl/torch-${PYTORCH}%2Bcu117/torch_scatter-2.1.2%2Bpt20cu117-cp310-cp310-linux_x86_64.whl
RUN git clone https://github.com/princeton-vl/DPVO.git && cd DPVO && git checkout 5833835 && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip && unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip && pip install -e .

RUN mkdir -p /models/checkpoints/ && pip install gdown==4.6.0 &&\
     gdown "https://drive.google.com/uc?id=1J6l8teyZrL0zFzHhzkC7efRhU0ZJ5G9Y&export=download&confirm=t" -O '/models/checkpoints/hmr2a.ckpt' &&\
     gdown "https://drive.google.com/uc?id=1kXTV4EYb-BI3H7J-bkR3Bc4gT9zfnHGT&export=download&confirm=t" -O '/models/checkpoints/dpvo.pth' &&\
     gdown "https://drive.google.com/uc?id=19qkI-a6xuwob9_RFNSPWf1yWErwVVlks&export=download&confirm=t" -O '/models/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar'

COPY ./ /WHAM/
RUN pip install -r /WHAM/requirements.txt
RUN ln -s /models/checkpoints/hmr2a.ckpt /WHAM/checkpoints/hmr2a.ckpt && ln -s /models/checkpoints/dpvo.pth /WHAM/checkpoints/dpvo.pth && ln -s /models/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar /WHAM/checkpoints/wham_vit_bedlam_w_3dpw.pth.tar

WORKDIR /WHAM/
