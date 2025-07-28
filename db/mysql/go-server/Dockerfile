# Multi-stage Docker build for MySQL server with LMDB
# This Dockerfile automatically handles all dependencies and builds

# Build stage
FROM golang:1.24-bullseye AS builder

# Install build dependencies for LMDB compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy go.mod and go.sum first for better caching
COPY go.mod go.sum ./

# Download Go dependencies
RUN go mod download

# Copy source code
COPY . .

# Install LMDB automatically using our script
RUN chmod +x scripts/install-lmdb.sh && \
    scripts/install-lmdb.sh

# Build the application (CGO enabled for LMDB)
RUN mkdir -p bin && \
    CGO_ENABLED=1 go build -o bin/mysql-server main.go && \
    CGO_ENABLED=1 go build -o bin/mysql-debug-server cmd/debug-server/main.go

# Runtime stage
FROM debian:bullseye-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r mysql && useradd -r -g mysql mysql

# Set working directory
WORKDIR /app

# Copy binaries from builder
COPY --from=builder /app/bin/mysql-server /app/bin/mysql-debug-server ./
COPY --from=builder /app/scripts/init.sql ./scripts/

# Copy LMDB shared libraries (needed for runtime)
COPY --from=builder /app/lmdb-lib/lib/liblmdb.so* /usr/local/lib/

# Update library cache so the runtime can find LMDB
RUN ldconfig

# Create data directory with proper permissions
RUN mkdir -p data && chown mysql:mysql data

# Switch to non-root user
USER mysql

# Expose MySQL port
EXPOSE 3306

# Health check to ensure server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD timeout 5 bash -c '</dev/tcp/localhost/3306' || exit 1

# Default command runs the main server
CMD ["./mysql-server"]

# Development stage (includes debug tools and mysql client)
FROM runtime AS development

# Switch back to root to install dev tools
USER root

# Install development and debugging tools
RUN apt-get update && apt-get install -y \
    default-mysql-client \
    netcat \
    procps \
    vim \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy source code for development debugging
COPY --from=builder /app /app-src

# Switch back to mysql user
USER mysql

# Default to debug server in development
CMD ["./mysql-debug-server"]

# Labels for image metadata
LABEL maintainer="MySQL Server with LMDB Team"
LABEL version="2.0"
LABEL description="MySQL-compatible server with LMDB persistent storage"
LABEL org.opencontainers.image.source="https://github.com/your-org/mysql-server-lmdb"