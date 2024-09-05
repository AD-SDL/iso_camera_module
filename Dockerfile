FROM ghcr.io/ad-sdl/wei

LABEL org.opencontainers.image.source=https://github.com/AD-SDL/mir_module
LABEL org.opencontainers.image.description="Drivers and REST API's for the mir plate handler robots"
LABEL org.opencontainers.image.licenses=MIT

#########################################
# Module specific logic goes below here #
#########################################

RUN mkdir -p mir_module

COPY ./src mir_module/src
COPY ./README.md mir_module/README.md
COPY ./pyproject.toml mir_module/pyproject.toml
COPY ./tests mir_module/tests

RUN --mount=type=cache,target=/root/.cache \
    pip install -e ./mir_module

CMD ["python", "mir_module/src/mir_rest_node.py"]

#########################################
