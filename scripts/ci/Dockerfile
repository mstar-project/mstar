FROM python:3.12-slim AS python

# The CPU attention tests use torch.compile, which builds C++ kernels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM python AS rust

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates cmake curl libzmq3-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
ENV PATH="/opt/cargo/bin:${PATH}"
# The repo's pinned toolchain, installed into the image so a container does not
# download it on first use (mac.sh passes the repo as the `repo` context).
COPY --from=repo rust-toolchain.toml /opt/rust-toolchain/
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh \
    && sh /tmp/rustup.sh -y --profile minimal --default-toolchain none \
    && rm /tmp/rustup.sh \
    && cd /opt/rust-toolchain && rustup toolchain install
