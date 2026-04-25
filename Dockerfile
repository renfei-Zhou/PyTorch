# Match the SAME version of Python and PyTorch （python --version | python -c "import torch; print(torch.__version__)"）
FROM python:3.12.9-slim

# 容器工作目录
WORKDIR /project

# ====================
# 德国 / 欧洲 官方源（速度飞快）
# ====================
RUN pip config set global.index-url https://pypi.org/simple
RUN pip install --upgrade pip

# 完全匹配你本地：PyTorch 2.9.1 CPU 版
RUN pip install torch==2.9.1+cpu torchvision==0.24.1+cpu --index-url https://download.pytorch.org/whl/cpu

# 安装你项目的所有依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# 德国时区（你在德国，用这个更准）
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

CMD ["bash"]