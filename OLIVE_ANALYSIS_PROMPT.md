# OLIVE 6.0.0 Technical Analysis Prompt

Use this prompt to analyze the OLIVE (Open Language Interface for Voice Exploitation) repository at https://github.com/rezker1/olive6.0.0

## Analysis Request

Please analyze the OLIVE 6.0.0 repository and provide a comprehensive technical overview covering:

### 1. Architecture Overview
- System architecture and design patterns
- Docker container structure (Martini setup)
- Plugin architecture and how plugins are loaded/executed
- Workflow engine design and JSON workflow structure
- Client-server communication protocols (protobuf definitions)

### 2. Core Capabilities
From the plugin directory structure and documentation, list all available capabilities:
- Speech processing (ASR, SAD, SID, etc.)
- Language processing (LID, LDD, translation)
- Security features (deepfake detection, speaker verification)
- Video processing (face detection/recognition)
- Text analytics (topic detection, keyword spotting)

### 3. API Analysis
- Python API (olivepy) structure and main classes
- Java API structure and usage patterns
- REST API endpoints from the Message Broker
- Protobuf message definitions (olive.proto, workflow.proto)

### 4. Workflow System
- How workflows are defined (JSON structure)
- Available workflow templates and their purposes
- How plugins are chained in workflows
- Input/output data flow between plugins

### 5. Plugin Development
- Plugin interface requirements (from plugin.py files)
- Configuration file formats (meta.conf, domain configs)
- How to extend OLIVE with new plugins
- Plugin lifecycle and execution model

### 6. Integration Points
- How to integrate OLIVE into existing systems
- Client implementation examples
- Streaming vs batch processing capabilities
- Multi-language support mechanisms

### 7. Deployment & Operations
- Martini management script capabilities
- GPU configuration for supported plugins
- TLS/SSL setup options
- Resource requirements and scaling considerations

### 8. Key Technical Insights
- Unique architectural decisions
- Performance optimization strategies visible in the code
- Security considerations in the design
- Extensibility patterns

Please focus on actionable technical details that would help someone:
- Integrate OLIVE into their application
- Develop new plugins or workflows
- Understand the system's capabilities and limitations
- Deploy and operate OLIVE in production

Note: The repository contains documentation and source code but not the trained models or Docker container. Focus on what can be learned from the available code and configuration files.