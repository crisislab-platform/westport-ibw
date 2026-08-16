# Self-contained CPU image for the computational-feasibility run.
#
# Everything the forecast reads is baked in -- weights, exposure vectors, ring
# cache, boundary conditions, and a synthetic sample raster carrying the real
# grid geometry (see stage_data.py). `docker run` therefore needs no volumes
# and no network, which is the whole point: hand someone the image and they can
# reproduce the timings on their own laptop.
#
# Build from the REPO ROOT, not from docker/:
#     python docker/stage_data.py
#     docker build -f docker/Dockerfile -t westport-ibw:cpu .
FROM mambaorg/micromamba:1.5.8

USER root
WORKDIR /app

# Solve the environment before the source goes in, so editing a module does not
# re-solve a ~2 GB environment on every rebuild.
COPY docker/environment.docker.yml /tmp/environment.docker.yml
RUN micromamba install -y -n base -f /tmp/environment.docker.yml \
    && micromamba clean --all --yes \
    && rm /tmp/environment.docker.yml

ENV PATH=/opt/conda/bin:$PATH

# The data root the image serves. modules/lib reads both of these, so the tree
# can sit anywhere; unset, they would resolve to the workstation's own paths.
ENV WESTPORT_DATA_ROOT=/data/westport \
    WESTPORT_REPO_ROOT=/app

# Threading is the one knob that decides the CPU timing. Left unset, torch and
# onnxruntime each spin up a pool sized to the visible core count and then
# contend with each other. run_experiment.sh sets these from --threads; these
# are only the defaults for a bare `docker run ... python main.py`.
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

COPY docker/data-bundle/out/    /data/westport/out/
COPY docker/data-bundle/data/   /data/westport/data/
COPY docker/data-bundle/repo/out/ /app/out/

COPY main.py /app/main.py
COPY modules/ /app/modules/
COPY docker/run_experiment.sh /app/run_experiment.sh
RUN chmod +x /app/run_experiment.sh

# Forecast outputs and the runtime_metrics.csv the experiment is read off.
# Declared so `docker run -v $PWD/results:/app/out/results` can lift them out.
RUN mkdir -p /data/westport/out/forecasts /app/out

ENTRYPOINT ["/app/run_experiment.sh"]
