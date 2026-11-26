# Task image of the quant-research environment: Python 3.12 with this
# project installed, so the reward's grader runs `python -m
# quant_research.grade` with the image's interpreter.
#
#   docker build -t quant-research:0.1.2 .
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
# Agent code runs as the sandbox user (`nobody`), which must not read the
# checks, the generator or the grader: from them it could learn the check
# thresholds and tune a lookahead to pass them. Only root, which runs the
# grader and the tools, reads this project's files; the sandbox user reads
# the modules its own process imports (`python -m quantenv.sandbox`) and,
# since Python's importer lists a package directory to find modules in it,
# the names of the other files in `quantenv`, not their contents.
RUN cd "$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
    && chmod -R o-rwx quantenv quant_research quant_research_notools \
    && chmod o+rx quantenv && chmod o+x quantenv/__pycache__ \
    && for m in __init__ alpha panel sandbox walkforward; do \
        chmod o+r quantenv/$m.py quantenv/__pycache__/$m.*.pyc; \
    done
WORKDIR /
