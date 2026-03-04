#!/usr/bin/env bash
# Generate gRPC Python stubs from proto files.
# Usage: bash scripts/generate_protos.sh

set -euo pipefail

PROTO_DIR="proto"
SERVICES=("embedding")

for service in "${SERVICES[@]}"; do
    OUT_DIR="services/${service}/src/generated"
    mkdir -p "$OUT_DIR"

    echo "Generating stubs for ${service}..."
    python3 -m grpc_tools.protoc \
        -I"$PROTO_DIR" \
        --python_out="$OUT_DIR" \
        --grpc_python_out="$OUT_DIR" \
        "$PROTO_DIR/${service}.proto"

    # Fix relative import in generated grpc file
    sed -i "s/import ${service}_pb2/from . import ${service}_pb2/" \
        "$OUT_DIR/${service}_pb2_grpc.py"

    # Create __init__.py
    touch "$OUT_DIR/__init__.py"

    echo "  -> ${OUT_DIR}/"
done

echo "Done."
