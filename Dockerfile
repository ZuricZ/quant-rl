# Task image of the quant-research environment: Python 3.12 with this
# project installed, so the reward's grader runs `python -m
# quant_research.grade` with the image's interpreter.
#
#   docker build -t quant-research:0.1.0 .
#
# Rebuild after changing quantenv/ or environments/: the grader runs the
# image's copy. For a remote runtime, push the image to a registry and set
# the task config's `image` to that reference.
ARG BASE=python:3.12-slim
FROM ${BASE}
WORKDIR /opt/quant-research
COPY pyproject.toml README.md ./
COPY quantenv ./quantenv
COPY environments ./environments
RUN pip install --no-cache-dir . && rm -rf /opt/quant-research
WORKDIR /
