# OLIVE 6.0.0 Martini

OLIVE (Open Language Interface for Voice Exploitation) is a modular AI platform for audio and speech processing.

## Quick Start

### Prerequisites
- Docker installed and running
- At least 8GB RAM allocated to Docker
- 20GB+ free disk space

### Installation

1. **Load the Docker container** (large file - 16GB):
   ```bash
   cd martini/
   docker load -i olive-martini-container-aiwdp-beta.rc3.tar
   ```
   Note: This operation can take 10-30 minutes depending on your system.

2. **Start the container**:
   ```bash
   ./martini.sh start
   ```
   
   With GPU support:
   ```bash
   ./martini.sh start --gpu
   ```

3. **Access the services**:
   - Raven Web UI: http://localhost:5580
   - Documentation: http://localhost:5570
   - OLIVE Server: localhost:5588

### Container Management

```bash
./martini.sh status   # Check container status
./martini.sh stop     # Stop the container
./martini.sh help     # View all commands
```

## Components

- **api/** - Python and Java OLIVE Client utilities
- **docs/** - OLIVE documentation
- **martini/** - Docker container and management scripts
- **OliveGUI/** - Nightingale Forensic GUI (optional)
- **oliveAppData/** - Plugins, workflows, and data directories

## Large Files

The Docker container file `olive-martini-container-aiwdp-beta.rc3.tar` (16GB) is tracked with Git LFS but may require additional storage quota on GitHub.

## License

See license.txt for details.