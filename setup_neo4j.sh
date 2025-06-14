#!/bin/bash

# Neo4j Setup Script for Linux/Windows (WSL)
# This script downloads, installs, and configures Neo4j Community Edition
# Supports Ubuntu/Debian, CentOS/RHEL/Fedora, and Windows with WSL

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NEO4J_VERSION="5.15.0"
NEO4J_HOME="/opt/neo4j"
NEO4J_USER="neo4j"
NEO4J_GROUP="neo4j"
SERVICE_NAME="neo4j"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS=$NAME
            VERSION=$VERSION_ID
        elif type lsb_release >/dev/null 2>&1; then
            OS=$(lsb_release -si)
            VERSION=$(lsb_release -sr)
        else
            OS=$(uname -s)
            VERSION=$(uname -r)
        fi
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        OS="Windows"
        VERSION="10"
    else
        OS=$(uname -s)
        VERSION=$(uname -r)
    fi
    
    print_status "Detected OS: $OS $VERSION"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. This is not recommended for production environments."
        SUDO=""
    else
        SUDO="sudo"
        print_status "Running as non-root user. Will use sudo when necessary."
    fi
}

# Function to install Java 17 (required for Neo4j 5.x)
install_java() {
    print_status "Checking Java installation..."
    
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d'.' -f1)
        if [ "$JAVA_VERSION" -ge 17 ]; then
            print_success "Java $JAVA_VERSION is already installed"
            return 0
        else
            print_warning "Java version $JAVA_VERSION found, but Neo4j requires Java 17+"
        fi
    fi
    
    print_status "Installing Java 17..."
    
    case "$OS" in
        "Ubuntu"*|"Debian"*)
            $SUDO apt update
            $SUDO apt install -y openjdk-17-jdk
            ;;
        "CentOS"*|"Red Hat"*|"Rocky"*|"AlmaLinux"*)
            $SUDO yum install -y java-17-openjdk java-17-openjdk-devel
            ;;
        "Fedora"*)
            $SUDO dnf install -y java-17-openjdk java-17-openjdk-devel
            ;;
        "Arch"*)
            $SUDO pacman -S --noconfirm jdk17-openjdk
            ;;
        *)
            print_error "Unsupported OS for automatic Java installation: $OS"
            print_status "Please install Java 17 manually and re-run this script"
            exit 1
            ;;
    esac
    
    # Set JAVA_HOME
    JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
    echo "export JAVA_HOME=$JAVA_HOME" | $SUDO tee -a /etc/environment
    export JAVA_HOME=$JAVA_HOME
    
    print_success "Java 17 installed successfully"
}

# Function to create Neo4j user and group
create_neo4j_user() {
    if ! getent group $NEO4J_GROUP >/dev/null 2>&1; then
        print_status "Creating group $NEO4J_GROUP..."
        $SUDO groupadd $NEO4J_GROUP
    fi
    
    if ! getent passwd $NEO4J_USER >/dev/null 2>&1; then
        print_status "Creating user $NEO4J_USER..."
        $SUDO useradd -r -g $NEO4J_GROUP -d $NEO4J_HOME -s /bin/bash $NEO4J_USER
    fi
}

# Function to download and install Neo4j
install_neo4j() {
    print_status "Downloading Neo4j Community Edition $NEO4J_VERSION..."
    
    # Create temporary directory
    TEMP_DIR=$(mktemp -d)
    cd $TEMP_DIR
    
    # Download Neo4j
    NEO4J_TARBALL="neo4j-community-$NEO4J_VERSION-unix.tar.gz"
    DOWNLOAD_URL="https://dist.neo4j.org/neo4j-community-$NEO4J_VERSION-unix.tar.gz"
    
    if command -v wget &> /dev/null; then
        wget $DOWNLOAD_URL
    elif command -v curl &> /dev/null; then
        curl -O $DOWNLOAD_URL
    else
        print_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    # Extract Neo4j
    print_status "Extracting Neo4j..."
    tar -xzf $NEO4J_TARBALL
    
    # Move to installation directory
    print_status "Installing Neo4j to $NEO4J_HOME..."
    $SUDO mkdir -p $(dirname $NEO4J_HOME)
    $SUDO mv neo4j-community-$NEO4J_VERSION $NEO4J_HOME
    
    # Set ownership
    $SUDO chown -R $NEO4J_USER:$NEO4J_GROUP $NEO4J_HOME
    
    # Clean up
    cd /
    rm -rf $TEMP_DIR
    
    print_success "Neo4j installed successfully"
}

# Function to configure Neo4j
configure_neo4j() {
    print_status "Configuring Neo4j..."
    
    # Backup original config
    $SUDO cp $NEO4J_HOME/conf/neo4j.conf $NEO4J_HOME/conf/neo4j.conf.backup
    
    # Configure Neo4j
    $SUDO tee $NEO4J_HOME/conf/neo4j.conf > /dev/null <<EOF
# Network connector configuration
server.default_listen_address=0.0.0.0
server.default_advertised_address=localhost

# HTTP Connector
server.http.enabled=true
server.http.listen_address=:7474
server.http.advertised_address=:7474

# HTTPS Connector
server.https.enabled=false

# Bolt connector
server.bolt.enabled=true
server.bolt.listen_address=:7687
server.bolt.advertised_address=:7687

# Directories
server.directories.data=data
server.directories.logs=logs
server.directories.lib=lib
server.directories.plugins=plugins
server.directories.import=import

# Memory settings
server.memory.heap.initial_size=512m
server.memory.heap.max_size=1G
server.memory.pagecache.size=512m

# Security
dbms.security.auth_enabled=true

# Allow upgrade from older versions
dbms.allow_upgrade=true

# Transaction timeout
dbms.transaction.timeout=60s

# Query timeout
dbms.transaction.query.timeout=0

# JVM additional settings
server.jvm.additional=-XX:+UseG1GC
server.jvm.additional=-XX:-OmitStackTraceInFastThrow
server.jvm.additional=-XX:+AlwaysPreTouch
server.jvm.additional=-XX:+UnlockExperimentalVMOptions
server.jvm.additional=-XX:+TrustFinalNonStaticFields
server.jvm.additional=-XX:+DisableExplicitGC
server.jvm.additional=-Djdk.tls.ephemeralDHKeySize=2048
server.jvm.additional=-Djdk.tls.rejectClientInitiatedRenegotiation=true
server.jvm.additional=-Dunsupported.dbms.udc.source=tarball
EOF

    # Set proper permissions
    $SUDO chown $NEO4J_USER:$NEO4J_GROUP $NEO4J_HOME/conf/neo4j.conf
    
    print_success "Neo4j configured successfully"
}

# Function to create systemd service
create_service() {
    print_status "Creating systemd service..."
    
    $SUDO tee /etc/systemd/system/$SERVICE_NAME.service > /dev/null <<EOF
[Unit]
Description=Neo4j Graph Database
After=network.target
Requires=network.target

[Service]
Type=forking
User=$NEO4J_USER
Group=$NEO4J_GROUP
ExecStart=$NEO4J_HOME/bin/neo4j start
ExecStop=$NEO4J_HOME/bin/neo4j stop
ExecReload=$NEO4J_HOME/bin/neo4j restart
TimeoutSec=120
Restart=on-failure
RestartSec=5
Environment="JAVA_HOME=$JAVA_HOME"
Environment="NEO4J_HOME=$NEO4J_HOME"
PIDFile=$NEO4J_HOME/run/neo4j.pid

[Install]
WantedBy=multi-user.target
EOF

    # Reload systemd and enable service
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable $SERVICE_NAME
    
    print_success "Systemd service created and enabled"
}

# Function to start Neo4j and set initial password
start_neo4j() {
    print_status "Starting Neo4j..."
    
    # Start Neo4j service
    $SUDO systemctl start $SERVICE_NAME
    
    # Wait for Neo4j to start
    print_status "Waiting for Neo4j to start..."
    for i in {1..30}; do
        if curl -s http://localhost:7474 >/dev/null 2>&1; then
            break
        fi
        echo -n "."
        sleep 2
    done
    echo
    
    if ! curl -s http://localhost:7474 >/dev/null 2>&1; then
        print_error "Neo4j failed to start. Check logs: $SUDO journalctl -u $SERVICE_NAME -f"
        exit 1
    fi
    
    print_success "Neo4j started successfully"
    
    # Set initial password
    print_status "Setting up initial password..."
    echo "Please set the initial password for the 'neo4j' user:"
    read -s -p "Enter new password: " NEW_PASSWORD
    echo
    read -s -p "Confirm password: " CONFIRM_PASSWORD
    echo
    
    if [ "$NEW_PASSWORD" != "$CONFIRM_PASSWORD" ]; then
        print_error "Passwords do not match!"
        exit 1
    fi
    
    # Change default password
    $SUDO -u $NEO4J_USER $NEO4J_HOME/bin/cypher-shell -u neo4j -p neo4j "ALTER USER neo4j SET PASSWORD '$NEW_PASSWORD'" 2>/dev/null || {
        print_warning "Could not change password automatically. Please change it manually using Neo4j Browser."
    }
    
    print_success "Password set successfully"
}

# Function to install useful tools
install_tools() {
    print_status "Installing useful tools..."
    
    case "$OS" in
        "Ubuntu"*|"Debian"*)
            $SUDO apt install -y curl wget unzip
            ;;
        "CentOS"*|"Red Hat"*|"Rocky"*|"AlmaLinux"*)
            $SUDO yum install -y curl wget unzip
            ;;
        "Fedora"*)
            $SUDO dnf install -y curl wget unzip
            ;;
        "Arch"*)
            $SUDO pacman -S --noconfirm curl wget unzip
            ;;
    esac
}

# Function to display final information
display_info() {
    print_success "Neo4j installation completed successfully!"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}Neo4j Information:${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${BLUE}Installation Directory:${NC} $NEO4J_HOME"
    echo -e "${BLUE}Version:${NC} $NEO4J_VERSION"
    echo -e "${BLUE}User:${NC} $NEO4J_USER"
    echo
    echo -e "${GREEN}Service Management:${NC}"
    echo "  Start:   sudo systemctl start neo4j"
    echo "  Stop:    sudo systemctl stop neo4j" 
    echo "  Restart: sudo systemctl restart neo4j"
    echo "  Status:  sudo systemctl status neo4j"
    echo "  Logs:    sudo journalctl -u neo4j -f"
    echo
    echo -e "${GREEN}Connection Information:${NC}"
    echo "  Web Interface:  http://localhost:7474"
    echo "  Bolt Protocol:  bolt://localhost:7687"
    echo "  Username:       neo4j"
    echo "  Password:       [The password you set]"
    echo
    echo -e "${GREEN}Configuration File:${NC}"
    echo "  Location: $NEO4J_HOME/conf/neo4j.conf"
    echo "  Backup:   $NEO4J_HOME/conf/neo4j.conf.backup"
    echo
    echo -e "${GREEN}Environment Variables for MCP:${NC}"
    echo "  NEO4J_URI=bolt://localhost:7687"
    echo "  NEO4J_USER=neo4j"
    echo "  NEO4J_PASSWORD=[your-password]"
    echo
    echo -e "${YELLOW}Next Steps:${NC}"
    echo "1. Open http://localhost:7474 in your browser"
    echo "2. Log in with username 'neo4j' and your password"
    echo "3. Configure your MCP server with the connection details above"
    echo "4. Test the connection using Neo4j Browser or cypher-shell"
    echo
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# Function to show help
show_help() {
    echo "Neo4j Setup Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  --version VERSION    Specify Neo4j version (default: $NEO4J_VERSION)"
    echo "  --home PATH         Specify installation directory (default: $NEO4J_HOME)"
    echo "  --user USERNAME     Specify Neo4j user (default: $NEO4J_USER)"
    echo "  --help              Show this help message"
    echo
    echo "Examples:"
    echo "  $0                           # Install with defaults"
    echo "  $0 --version 5.14.0         # Install specific version"
    echo "  $0 --home /usr/local/neo4j   # Install to custom directory"
    echo
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            NEO4J_VERSION="$2"
            shift 2
            ;;
        --home)
            NEO4J_HOME="$2"
            shift 2
            ;;
        --user)
            NEO4J_USER="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${GREEN}Neo4j Community Edition Setup Script${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo
    
    # Check if systemd is available
    if ! command -v systemctl &> /dev/null; then
        print_error "This script requires systemd. Please install Neo4j manually for your system."
        exit 1
    fi
    
    detect_os
    check_root
    install_tools
    install_java
    create_neo4j_user
    install_neo4j
    configure_neo4j
    create_service
    start_neo4j
    display_info
}

# Run main function
main "$@"