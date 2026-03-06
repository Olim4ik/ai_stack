FROM python:3.12-slim

WORKDIR /app

COPY services/mcp_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/retrieval.proto /app/proto/retrieval.proto

# Generate gRPC stubs for knowledge_search tool
RUN mkdir -p /app/src/generated && \
    python -m grpc_tools.protoc \
        -I/app/proto \
        --python_out=/app/src/generated \
        --grpc_python_out=/app/src/generated \
        /app/proto/retrieval.proto && \
    touch /app/src/generated/__init__.py && \
    sed -i 's/import retrieval_pb2/from . import retrieval_pb2/' /app/src/generated/retrieval_pb2_grpc.py

COPY services/mcp_server/src/ /app/src/

CMD ["python", "-m", "src.main"]
