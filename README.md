# Unified Code Context MCP Server

A powerful Model Context Protocol (MCP) server that analyzes your codebase and builds a comprehensive knowledge graph using Neo4j. Automatically discover imports, add code to a graph database, and find relevant code snippets with intelligent search capabilities.

## ✨ Features

- **Import Analysis** - Extract and analyze package imports from Python, JavaScript, TypeScript, and Java files
- **Graph Database** - Build comprehensive code graphs in Neo4j with dependency tracking
- **Auto Package Discovery** - Automatically find and add Python packages (standard library + pip-installed)
- **Background Processing** - Non-blocking operations with real-time progress tracking
- **Smart Code Search** - Find functions, classes, and content with relevance scoring
- **Job Management** - Track neo4j query jobs with time estimation and progress monitoring

## 🚀 Quick Start

### 1. Install Neo4j / Use a Cloud Service

Use the provided setup script for automatic installation:

```bash
# Make the script executable
chmod +x setup_neo4j.sh

# Install Neo4j with default settings
./setup_neo4j.sh

# Or with custom options
./setup_neo4j.sh --version 5.15.0 --home /opt/neo4j
```

The script will:
- Install Java 17 (if needed)
- Download and configure Neo4j Community Edition
- Create systemd service for auto-startup
- Set up secure authentication

### 2. Configure MCP Server

Add to your MCP configuration file:

```json
{
  "mcpServers": {
    "unified-code-context-server": {
      "command": "python",
      "args": ["path/to/Naive_mcp.py"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your-password-here"
      }
    }
  }
}
```

### 3. Install Python Dependencies

```bash
pip install neo4j requests
```

## 🛠️ Available Tools

### `list_imports`
Extract all package imports from code files
```json
{
  "path": "./my-project",
  "language": "python",
  "recursive": true
}
```

### `add_code_to_graph`
Add local code to the graph database (background processing)
```json
{
  "path": "./my-project",
  "is_dependency": false
}
```

### `add_package_to_graph`
Auto-discover and add Python packages to the graph
```json
{
  "package_name": "subprocess",
  "is_dependency": true
}
```

### `check_job_status`
Monitor background processing jobs
```json
{
  "job_id": "abc123-def456"
}
```

### `list_jobs`
View all processing jobs and their status

### `find_code`
Search for relevant code snippets
```json
{
  "query": "authentication"
}
```

## 📋 Example Workflow

1. **Add your project to the graph:**
   ```bash
   add_code_to_graph: {"path": "./my-app", "is_dependency": false}
   # Returns: {"job_id": "abc123", "estimated_duration": "45s"}
   ```

2. **Add dependencies:**
   ```bash
   add_package_to_graph: {"package_name": "requests"}
   add_package_to_graph: {"package_name": "flask"}
   ```

3. **Monitor progress:**
   ```bash
   check_job_status: {"job_id": "abc123"}
   # Returns: {"progress_percentage": 67.3, "current_file": "app.py"}
   ```

4. **Search your codebase:**
   ```bash
   find_code: {"query": "user authentication"}
   # Returns ranked results from functions, classes, and content
   ```

## 🔧 Neo4j Management

Control Neo4j service:
```bash
# Start/stop/restart
sudo systemctl start neo4j
sudo systemctl stop neo4j
sudo systemctl restart neo4j

# Check status and logs
sudo systemctl status neo4j
sudo journalctl -u neo4j -f
```

Access Neo4j Browser: http://localhost:7474

## 🏗️ Architecture

The server creates a rich graph structure in Neo4j:

- **Repository** nodes for each codebase
- **File** nodes for individual source files
- **Function** nodes with metadata (args, docstrings, line numbers)
- **Class** nodes with inheritance information
- **Variable** nodes with context and values
- **Module** nodes for import relationships

All nodes include `is_dependency` flags to distinguish between your code and external dependencies.

## 📊 Progress Tracking

Background jobs provide detailed progress information:
- File count estimation
- Processing time estimation
- Real-time progress percentage
- Current file being processed
- Error tracking and reporting

## 🔍 Search Capabilities

The search system uses multiple strategies:
- **Function name matching** (exact and fuzzy)
- **Class name matching** 
- **Content search** (source code and docstrings)
- **Relevance scoring** (prioritizes your code over dependencies)

## 🐛 Troubleshooting

**Neo4j Connection Issues:**
- Check if Neo4j is running: `sudo systemctl status neo4j`
- Verify credentials in environment variables
- Check Neo4j logs: `sudo journalctl -u neo4j -f`

**Job Processing Issues:**
- Check debug logs: `~/mcp_debug.log`
- Monitor job status with `check_job_status`
- Use `list_jobs` to see all processing history

## 📝 Requirements

- Python 3.8+
- Neo4j 5.x
- Java 17+ (automatically installed by setup script)
- Linux system with systemd (or Windows with WSL)

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the codebase analysis capabilities!

## 📄 License

GNU AFFERO GENERAL PUBLIC LICENSE - feel free to use as needed.