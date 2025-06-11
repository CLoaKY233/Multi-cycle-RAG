# Installation Guide

This guide provides comprehensive instructions for setting up the Reflexion RAG Engine in different environments.

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.13 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for installation, additional space for documents
- **Network**: Internet connection for model APIs and SurrealDB cloud

### Recommended Requirements
- **RAM**: 16GB or higher for large document collections
- **CPU**: 4+ cores for optimal performance
- **Storage**: SSD for better I/O performance
- **Network**: Stable broadband connection (50+ Mbps)

### Dependencies
- **GitHub Personal Access Token** with Models access
- **SurrealDB** instance (local or cloud)
- **UV Package Manager** (recommended) or pip

## Prerequisites Setup

### 1. Install Python 3.13+

#### Windows
```bash
# Download from python.org or use winget
winget install Python.Python.3.13
```

#### macOS
```bash
# Using Homebrew
brew install python@3.13

# Or download from python.org
```

#### Linux (Ubuntu/Debian)
```bash
# Add deadsnakes PPA for latest Python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev
```

### 2. Install UV Package Manager (Recommended)

UV is a fast Python package installer and resolver, written in Rust.

#### Installation
```bash
# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pip
pip install uv
```

#### Verify Installation
```bash
uv --version
# Should output: uv 0.x.x
```

### 3. GitHub Personal Access Token

1. Go to [GitHub Settings > Developer settings > Personal access tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Select the following scopes:
   - `repo` (Full control of private repositories)
   - `read:org` (Read org and team membership)
4. Copy the generated token and save it securely

### 4. SurrealDB Setup

#### Option A: SurrealDB Cloud (Recommended)
1. Visit [SurrealDB Cloud](https://surrealdb.com/cloud)
2. Create a free account
3. Create a new database instance
4. Note the connection URL, namespace, database, username, and password

#### Option B: Local SurrealDB
```bash
# Install SurrealDB
# macOS
brew install surrealdb/tap/surreal

# Linux
curl --proto '=https' --tlsv1.2 -sSf https://install.surrealdb.com | sh

# Windows
# Download from https://surrealdb.com/install

# Start local instance
surreal start --log trace --user root --pass root memory
```

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone https://github.com/cloaky233/rag_new.git
cd rag_new

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies
uv sync

# Verify installation
uv run rag.py --help
```

### Method 2: Manual Install with pip

```bash
# Clone the repository
git clone https://github.com/cloaky233/rag_new.git
cd rag_new

# Create virtual environment
python3.13 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify installation
python rag.py --help
```

### Method 3: Development Install

```bash
# Clone the repository
git clone https://github.com/cloaky233/rag_new.git
cd rag_new

# Create virtual environment
uv venv
source .venv/bin/activate

# Install in development mode
uv sync --dev

# Install pre-commit hooks
pre-commit install

# Run tests to verify
pytest tests/
```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
# Copy example configuration
cp .env.example .env

# Edit the configuration
nano .env  # or your preferred editor
```

### 2. Required Configuration

```bash
# GitHub Models Configuration
GITHUB_TOKEN=your_github_pat_token_here
LLM_MODEL=meta/Meta-Llama-3.1-405B-Instruct
EVALUATION_MODEL=cohere/Cohere-command-r
SUMMARY_MODEL=meta/Meta-Llama-3.1-70B-Instruct

# Azure AI Inference Embeddings
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_ENDPOINT=https://models.inference.ai.azure.com
EMBEDDING_PROVIDER=azure_ai
EMBEDDING_BATCH_SIZE=100

# SurrealDB Configuration
SURREALDB_URL=wss://your-surreal-instance.surreal.cloud
SURREALDB_NS=rag
SURREALDB_DB=rag
SURREALDB_USER=your_username
SURREALDB_PASS=your_password

# Reflexion Settings
ENABLE_REFLEXION_LOOP=true
MAX_REFLEXION_CYCLES=5
CONFIDENCE_THRESHOLD=0.8
INITIAL_RETRIEVAL_K=3
REFLEXION_RETRIEVAL_K=5

# Performance Settings
ENABLE_MEMORY_CACHE=true
MAX_CACHE_SIZE=1000
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### 3. Configuration Validation

```bash
# Validate configuration
uv run rag.py config

# Test connections
uv run python -c "
from src.config.settings import settings
print('Configuration loaded successfully!')
print(f'LLM Model: {settings.llm_model}')
print(f'SurrealDB URL: {settings.surrealdb_url}')
"
```

## Initial Setup

### 1. Test Installation

```bash
# Check system info
uv run rag.py config

# Test basic functionality
uv run python -c "
import asyncio
from src.rag.engine import RAGEngine

async def test():
    engine = RAGEngine()
    info = engine.get_engine_info()
    print('Engine initialized successfully!')
    print(f'Engine type: {info[\"engine_type\"]}')

asyncio.run(test())
"
```

### 2. Prepare Documents

```bash
# Create documents directory
mkdir -p docs

# Add your documents to the docs directory
# Supported formats: PDF, TXT, DOCX, MD, HTML
cp /path/to/your/documents/* docs/
```

### 3. Initial Document Ingestion

```bash
# Ingest documents
uv run rag.py ingest --docs_path=./docs

# Verify ingestion
uv run rag.py config
# Should show document count > 0
```

### 4. Test Chat Interface

```bash
# Start interactive chat
uv run rag.py chat

# Try a simple query
Query: What is the main topic of the documents?

# Try a complex query to test reflexion
Query: Provide a comprehensive analysis comparing the different approaches mentioned in the documents, including their advantages and limitations.
```

## Advanced Installation Options

### Docker Installation

```bash
# Build Docker image
docker build -t reflexion-rag .

# Run with environment file
docker run --env-file .env -v $(pwd)/docs:/app/docs reflexion-rag

# Or use docker-compose
docker-compose up -d
```

### Kubernetes Deployment

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -l app=reflexion-rag

# Access logs
kubectl logs -f deployment/reflexion-rag
```

### Cloud Deployment

#### AWS
```bash
# Using AWS CDK
cdk deploy --all

# Or using CloudFormation
aws cloudformation create-stack --stack-name reflexion-rag --template-body file://aws/cloudformation.yaml
```

#### Google Cloud
```bash
# Deploy to Cloud Run
gcloud run deploy reflexion-rag --source .

# Or use Kubernetes Engine
gcloud container clusters create reflexion-rag-cluster
kubectl apply -f k8s/
```

#### Azure
```bash
# Deploy to Container Instances
az container create --resource-group myResourceGroup --name reflexion-rag --image reflexion-rag:latest
```

## Troubleshooting

### Common Issues

#### 1. Python Version Issues
```bash
# Check Python version
python --version
python3 --version

# Install specific version if needed
pyenv install 3.13.0
pyenv global 3.13.0
```

#### 2. Permission Issues (Linux/macOS)
```bash
# Fix permissions
chmod +x rag.py

# Or run with python explicitly
python rag.py chat
```

#### 3. Package Installation Issues
```bash
# Clear package cache
uv cache clean

# Reinstall dependencies
rm -rf .venv
uv venv
uv sync

# Alternative: use pip
pip install --force-reinstall -r requirements.txt
```

#### 4. SurrealDB Connection Issues
```bash
# Test connection manually
surreal sql --conn wss://your-instance.surreal.cloud --user your_user --pass your_pass --ns rag --db rag

# Check connection in Python
python -c "
import asyncio
from src.vectorstore.surrealdb_store import SurrealDBVectorStore

async def test():
    store = SurrealDBVectorStore()
    await store.connect()
    print('Connected successfully!')

asyncio.run(test())
"
```

#### 5. GitHub Token Issues
```bash
# Test token validity
curl -H "Authorization: token YOUR_GITHUB_TOKEN" https://api.github.com/user

# Test GitHub Models access
curl -H "Authorization: Bearer YOUR_GITHUB_TOKEN" https://models.inference.ai.azure.com/models
```

### Performance Issues

#### 1. Slow Response Times
- Increase `EMBEDDING_BATCH_SIZE` for faster ingestion
- Enable memory cache: `ENABLE_MEMORY_CACHE=true`
- Use local SurrealDB for faster queries
- Reduce `MAX_REFLEXION_CYCLES` for faster responses

#### 2. Memory Issues
- Reduce `MAX_CACHE_SIZE`
- Decrease `CHUNK_SIZE` for smaller memory footprint
- Use swap space if needed
- Monitor memory usage with `htop` or `Activity Monitor`

#### 3. Network Issues
- Check internet connectivity
- Verify SurrealDB URL accessibility
- Test GitHub Models API availability
- Consider using local models for offline operation

### Debugging

#### Enable Debug Logging
```bash
# Set environment variables
export LOG_LEVEL=DEBUG
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run with verbose output
uv run rag.py chat --verbose
```

#### Check System Resources
```bash
# Monitor system resources
htop  # Linux/macOS
# or use Task Manager on Windows

# Check disk space
df -h  # Linux/macOS
dir  # Windows

# Monitor network
netstat -an | grep 8000  # Check if ports are in use
```

## Verification

### Installation Verification Checklist

- [ ] Python 3.13+ installed and accessible
- [ ] UV package manager installed
- [ ] Repository cloned successfully
- [ ] Virtual environment created
- [ ] Dependencies installed without errors
- [ ] Environment variables configured
- [ ] SurrealDB connection working
- [ ] GitHub token valid and has Models access
- [ ] Documents ingested successfully
- [ ] Chat interface responds to queries
- [ ] Reflexion cycles complete successfully
- [ ] Memory cache functioning (if enabled)
- [ ] Performance within acceptable ranges

### Health Check Script

```bash
# Run comprehensive health check
uv run python scripts/health_check.py

# Or manual verification
uv run rag.py config
uv run rag.py ingest --docs_path=./docs
uv run python -c "
import asyncio
from src.rag.engine import RAGEngine

async def test():
    engine = RAGEngine()
    response = await engine.query_stream('Test query')
    async for chunk in response:
        print(chunk.content, end='')
        break  # Just test first chunk

asyncio.run(test())
"
```

## Next Steps

After successful installation:

1. **Read the [Configuration Guide](configuration.md)** for advanced settings
2. **Explore the [API Documentation](api.md)** for programmatic usage
3. **Check the [Performance Tuning Guide](performance.md)** for optimization
4. **Review the [Architecture Documentation](architecture.md)** to understand the system
5. **Join the community** on [GitHub Discussions](https://github.com/cloaky233/rag_new/discussions)

## Support

If you encounter issues during installation:

1. **Check the [Troubleshooting Guide](troubleshooting.md)**
2. **Search existing [GitHub Issues](https://github.com/cloaky233/rag_new/issues)**
3. **Create a new issue** with detailed error information
4. **Join the [Discord community](https://discord.gg/reflexion-rag)** for real-time help
5. **Contact the author**: [laysheth1@gmail.com](mailto:laysheth1@gmail.com)

---

*Installation guide last updated: June 2025*