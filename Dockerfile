# syntax=docker/dockerfile:1.6
FROM python:3.12-slim AS base

# System deps: only what chromadb + sqlite need. Slim down to keep image small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user.
RUN groupadd --system agent && useradd --system --gid agent --create-home --home-dir /home/agent agent

WORKDIR /app

# Layer cache: install deps first so source changes don't reinstall everything.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application.
COPY agent/ ./agent/
COPY cli.py ./
COPY demo/ ./demo/
COPY README.md docs/ ./

# Sessions (JSON files, traces, chroma local storage) live here.
# Mount a host volume to /app/sessions for persistence.
RUN mkdir -p /app/sessions && chown -R agent:agent /app
ENV SESSIONS_DIR=/app/sessions
USER agent

# tini = clean PID 1 signal handling; crucial for Ctrl-C in `docker run -it`.
# CMD is empty so `docker compose run --rm agent` invokes `python cli.py` with
# no args, which lets the auto-naming flow pick a Chinese slug after the
# first turn (otherwise CMD would default to --help and the container would
# print usage and exit immediately). Override per-invocation:
#   docker compose run --rm agent --session weather
#   docker compose run --rm agent --once "what is 2+2?"
ENTRYPOINT ["/usr/bin/tini", "--", "python", "cli.py"]
CMD []
