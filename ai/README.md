# OpenShield RAG Pipeline

RAG pipeline for OpenShield rules and compliance frameworks.

## Usage

python -m ai.pipeline build
python -m ai.pipeline refresh
python -m ai.pipeline query "What is Network Watcher?"

## API Endpoints

GET /api/ai/query?q=text
POST /api/ai/refresh
GET /api/ai/status
