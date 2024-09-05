FROM ghcr.io/ad-sdl/wei

LABEL org.opencontainers.image.source=https://github.com/AD-SDL/rpl_tag_engine_module
LABEL org.opencontainers.image.description="Drivers and REST API's for the RPL tag engine and camera."
LABEL org.opencontainers.image.licenses=MIT

#########################################
# Module specific logic goes below here #
#########################################

RUN mkdir -p rpl_tag_engine_module

COPY ./src rpl_tag_engine_module/src
COPY ./README.md rpl_tag_engine_module/README.md
COPY ./pyproject.toml rpl_tag_engine_module/pyproject.toml
COPY ./tests rpl_tag_engine_module/tests

RUN --mount=type=cache,target=/root/.cache \
    pip install -e ./rpl_tag_engine_module

CMD ["python", "rpl_tag_engine_module/src/rpl_tag_engine_rest_node.py"]

#########################################
