#!/usr/bin/env bash
docker run \
    --gpus=all \
    --interactive \
    --tty \
    --volume "<repo location>:/home/mambauser/dev-container" \
    --user mambauser \
    -e DISPLAY=:0 \
    -e LD_LIBRARY_PATH=/opt/conda/lib \
    -e LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7 \
    -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
    pcad-dataset-runner \
    bash
