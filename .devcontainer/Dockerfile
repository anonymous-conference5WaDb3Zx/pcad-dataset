FROM mambaorg/micromamba:1.4.4-focal-cuda-11.7.1
# ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY --from=tianon/gosu /usr/local/bin/gosu /usr/local/bin/gosu
USER root
ENV USERNAME=mambauser

COPY --chown=$MAMBA_USER:$MAMBA_USER env-pre.yaml /tmp/env-pre.yaml
RUN --mount=type=cache,mode=0777,uid=1000,gid=1000,target=/opt/conda/pkgs micromamba install -y -f /tmp/env-pre.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN --mount=type=cache,mode=0777,uid=1000,gid=1000,target=/opt/conda/pkgs \
    CONDA_VERBOSITY=-2 micromamba install -n base -y -f /tmp/env.yaml

COPY --chown=$MAMBA_USER:$MAMBA_USER env-pre.yaml /tmp/env-pre.yaml
RUN --mount=type=cache,mode=0777,uid=1000,gid=1000,target=/opt/conda/pkgs micromamba create -n blender -y -f /tmp/env-pre.yaml
COPY --chown=$MAMBA_USER:$MAMBA_USER env-blender.yaml /tmp/env-blender.yaml
RUN --mount=type=cache,mode=0777,uid=1000,gid=1000,target=/opt/conda/pkgs micromamba install -n blender -y -f /tmp/env-blender.yaml

USER $MAMBA_USER

WORKDIR /home/mambauser/dev-container
USER root
RUN rm /etc/apt/sources.list.d/cuda.list
RUN apt-get update

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN --mount=type=cache,target=/var/cache/apt apt-get install -y tzdata wget

RUN --mount=type=cache,target=/var/cache/apt apt-get install -y \
        libxrender1 \
        libxrender-dev \
        libffi-dev \
        libglew-dev \
        libssl-dev \
        libx11-dev \
        libxcursor-dev \
        libxi-dev \
        libxinerama-dev \
        libxrandr-dev \
        libxxf86vm-dev \
        zlib1g-dev \
        libxkbcommon-dev \
        libsm6 \
        libxext6 \
        xvfb \
        libgomp1

RUN --mount=type=cache,target=/var/cache/apt apt-get install -y git

# scipy does not link to python environment --> need to enforce
# directly import through LD_PRELOAD
# RUN echo $(unlink /usr/lib/x86_64-linux-gnu/libstdc++.so.6)
# RUN echo $(cp /opt/conda/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6)
RUN echo $(mkdir /usr/lib/dri/)
RUN echo $(cp /usr/lib/x86_64-linux-gnu/dri/swrast_dri.so /usr/lib/dri/swrast_dri.so)

# Get the xz utilities and unzip to decompress the Blender distribution and addons
RUN --mount=type=cache,target=/var/cache/apt apt-get install -y xz-utils unzip

# install blender
COPY --chown=root:root blender-3.5.1-linux-x64.tar.xz /tmp/downloads/
# RUN --mount=type=cache,mode=0777,uid=0,gid=0,target=/tmp/downloads wget -P /tmp/downloads https://download.blender.org/release/Blender3.5/blender-3.5.1-linux-x64.tar.xz
RUN tar -xvf /tmp/downloads/blender-3.5.1-linux-x64.tar.xz -C /usr/local/
ENV PATH /usr/local/blender-3.5.1-linux-x64:$PATH

# install eyek addon for blender
COPY --chown=root:root eyek_addon.zip /tmp/downloads/
# RUN --mount=type=cache,mode=0777,uid=0,gid=0,target=/tmp/downloads wget -P /tmp/downloads https://github.com/p4ymak/eyek/releases/download/v0.2.7/eyek_addon.zip
RUN unzip /tmp/downloads/eyek_addon.zip -d /usr/local/blender-3.5.1-linux-x64/3.5/scripts/addons/
RUN chown mambauser:mambauser -R /usr/local/

COPY --chown=root:root id_rsa_dvc_remote /root/.ssh/id_rsa_dvc_remote
COPY --chown=root:root ssh-config /root/.ssh/config
USER $MAMBA_USER

COPY --chown=$MAMBA_USER:$MAMBA_USER blenderproc_initializer.py /home/mambauser/blenderproc_initializer.py

# USER root
# RUN --mount=type=cache,target=/var/cache/apt apt-get install -y libglib2.0-0
# USER $MAMBAUSER

# hackerman says: download kernes for blenderproc here
# RUN /bin/bash -c "eval $(micromamba shell hook --shell=bash) && source /usr/local/bin/_activate_current_env.sh && blenderproc run /home/mambauser/blenderproc_initializer.py"

# TODO: Remove gcc and future-strings after installation process has finished
WORKDIR /home/mambauser/dev-container
