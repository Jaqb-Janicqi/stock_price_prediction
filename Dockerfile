FROM debian:bookworm-slim

# Install necessary tools
RUN apt-get update && \
    apt-get install -y curl bzip2 \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba (amd64)
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
RUN mv bin/micromamba /usr/local/bin/

WORKDIR /app
ENV MAMBA_ROOT_PREFIX /root/micromamba

# Create micromamba environment
COPY environment.yml /app/environment.yml
RUN micromamba create -n env -f environment.yml -y

# Install the CPU-only version of PyTorch and related packages
# RUN /usr/local/bin/micromamba run -n env pip install torch==2.1.1 torchvision==0.16.1 torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install pytorch-lightning
# RUN /usr/local/bin/micromamba run -n env pip install pytorch-lightning

# Ensure the environment is activated
RUN echo 'eval "$(micromamba shell hook --shell bash)"' >> /app/bash-profile && \
    echo 'micromamba activate env' >> /app/bash-profile
RUN cat /app/bash-profile >> /etc/bash.bashrc
RUN cat /app/bash-profile >> /root/.bashrc

ENV PATH /root/micromamba/envs/env/bin:$PATH

COPY . /app

ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
