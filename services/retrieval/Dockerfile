FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ../../proto/retrieval.proto /app/proto/retrieval.proto
COPY ../../proto/embedding.proto /app/proto/embedding.proto

# Generate gRPC stubs
RUN mkdir -p /app/src/generated && \
    python -m grpc_tools.protoc \
        -I/app/proto \
        --python_out=/app/src/generated \
        --grpc_python_out=/app/src/generated \
        /app/proto/retrieval.proto /app/proto/embedding.proto && \
    touch /app/src/generated/__init__.py && \
    sed -i 's/import retrieval_pb2/from . import retrieval_pb2/' /app/src/generated/retrieval_pb2_grpc.py && \
    sed -i 's/import embedding_pb2/from . import embedding_pb2/' /app/src/generated/embedding_pb2_grpc.py

COPY src/ /app/src/

EXPOSE 50051

CMD ["python", "-m", "src.main"]
