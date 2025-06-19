#!/bin/bash
# Script to generate Python gRPC code from fleet_manager.proto
# Usage: bash grpc_gen.sh

set -e

PROTO_DIR="$(dirname "$0")/../proto"
OUT_DIR="$PROTO_DIR"

python3 -m grpc_tools.protoc \
  -I"$PROTO_DIR" \
  --python_out="$OUT_DIR" \
  --grpc_python_out="$OUT_DIR" \
  "$PROTO_DIR/fleet_manager.proto"

echo "gRPC Python code generated in $OUT_DIR from $PROTO_DIR/fleet_manager.proto"
