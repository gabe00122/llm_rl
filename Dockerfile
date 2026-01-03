# ---------------------------------------------------
# Stage 1: Builder (Compiles Rust & installs deps)
# ---------------------------------------------------
FROM python:3.13-slim-bookworm AS builder

# Copy uv binary (Efficient, no curl/install needed)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install build tools (GCC, Rust, etc.)
RUN apt update && apt install -y build-essential curl

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Enable bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy dependency definitions first (Caching layer)
COPY README.md pyproject.toml uv.lock Cargo.toml Cargo.lock pyrefly.toml ./

# Sync dependencies (creates .venv)
# We tell uv to use the system python to avoid downloading another copy
RUN uv sync --frozen --no-install-project --python /usr/local/bin/python

# Copy source code and compile the project (Rust extensions)
COPY src src
COPY python python
RUN uv sync --frozen --python /usr/local/bin/python

# ---------------------------------------------------
# Stage 2: Runtime (Minimal image)
# ---------------------------------------------------
FROM python:3.13-slim-bookworm

WORKDIR /app

# Copy ONLY the virtual environment and necessary configs from builder
# We do NOT copy Rust, GCC, or source code (unless strictly required by the app)
COPY --from=builder /app/.venv /app/.venv
COPY configs configs
COPY start.sh /start.sh

# Fix permissions
RUN chmod +x /start.sh

# Add virtual environment to PATH
# This means we don't need to use `uv run`, just standard `python` or commands work
ENV PATH="/app/.venv/bin:$PATH"

CMD ["/start.sh"]