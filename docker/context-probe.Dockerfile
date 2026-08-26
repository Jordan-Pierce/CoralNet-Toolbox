# Prints exactly what .dockerignore admits into the build context.
# Not part of the app image -- a check you can re-run any time:
#
#   docker build -f docker/context-probe.Dockerfile --progress=plain -t ctx-probe .
#
FROM busybox
COPY . /ctx
RUN echo "=== CONTEXT SIZE ===" && du -sh /ctx && \
    echo "=== TOP LEVEL ===" && ls -A /ctx && \
    echo "=== WEIGHTS / CACHES THAT SLIPPED IN (want: none) ===" && \
    { find /ctx \( -name '*.pt' -o -name '*.pth' -o -name '*.onnx' \
         -o -name '*.engine' -o -name '__pycache__' \) | head -20; } && \
    echo "=== 10 LARGEST FILES ===" && \
    find /ctx -type f -exec du -k {} + | sort -rn | head -10
