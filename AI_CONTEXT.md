# AiiDA-Worker Architecture & Code Review Context

## Overview
This document serves as the context transfer for AI agents working across different machines, preserving the review of the AiiDA-Worker ("The Body") architecture. The worker operates entirely as an execution and data-retrieval engine for SABR. It is the sole owner of the AiiDA environment and database.

## 1. Core Architecture: Brain-Body Split
AiiDA-Worker exposes the state of the local AiiDA profile via deterministic REST endpoints. SABR interacts with this API to query provenance, submit processes, and execute autonomous scripts. The FastAPI backend correctly injects AiiDA modules dynamically to ensure profile loads are respected.

## 2. Key Design Patterns Analyzed

### Registry Pattern (Persistent Skills)
**Location:** `routers/registry.py`, `core/scripts.py`
**Implementation:** A powerful feature allowing SABR to register raw Python scripts to the file system (via `register_script`). These scripts store metadata (`script_meta_path`) and source logic. When executed via `/execution/execute/{script_name}`, the worker dynamically imports them (`importlib.util`), injects parameters, and awaits their `main(params)` entry points, preventing the need to pass massive code strings repeatedly over the network.

### Execution Sandbox (Arbitrary Code Run)
**Location:** `routers/execution.py` (`_execute_python_script`)
**Implementation:** A raw standard python execution sandbox is available at `/management/run-python`. It creates a local `exec()` dictionary containing `orm`, `plugins`, and `engine` to give SABR full, unhindered access to the AiiDA Core API, capturing trailing output via `io.StringIO()` buffering.

### Data Aggregation (Body side of Presentation)
**Location:** `routers/process.py`, `core/node_utils.py`, `core/process_utils.py`
**Implementation:** Endpoints like `/process/{identifier}` implement heavy lifting (such as using `defaultdict` for counting identical labels) and use `ProcessTree` to iteratively compile the node structure recursively (`to_dict`). 
- **Provenenance Links (Verbose):** Uses `base.links.get_incoming()` and `base.links.get_outgoing()` to capture the full provenance graph.
- **Direct Links (Concise):** Uses `node.inputs._get_keys()` and `node.inputs._get_node_by_link_label()` to extract only the explicit port mappings, avoiding over-fetching of hidden subprocess inputs in WorkChains.
- **Log Parsing:** Logs are parsed efficiently using `QueryBuilder().append(orm.Log...`. This minimizes the payload footprint before sending it over the bridge to SABR's Presenters.

## Technology Stack
- **FastAPI / Pydantic:** Core structural API mapping.
- **AiiDA Core:** Direct data sourcing.
- **Worker File System:** Persistent storage of "skills" and memory payloads.
- **Analysis Library:** Standardized scientific analytical modules.

## 4. AiiDA Analysis Library (Modular Logic)
**Location:** `repository/analysis/`
**Implementation:** A structured library designed to replace fragmented Jupyter Notebooks. It decouples core scientific logic from AiiDA's infrastructure and plotting.
- **Common Utilities (`common_utils.py`):** Centralized physical formulas (e.g., elasticity averages, ratio calculations) shared across all analysis modules.
- **Domain Modules (e.g., `born_charges/`):** Encapsulated logic for specific scientific purposes. Modules like `BornAnalyzer` extract data from AiiDA WorkChain outputs, compute derivative properties, and check stability.
- **Standardized Entry Points (`entries/`):** Uniform Python scripts (e.g., `run_born.py`) implementing a `run(node_pk, **kwargs)` function. These serve as the primary interface for SABR and other external services, returning JSON-serializable results.

## 3. Serialization Insights
### AiiDA Node Serialization (`to_jsonable`)
**Location:** `core/utils.py`
**Implementation:** A utility function used across all API responses to safely cast complex components to standard JSON. Historically, it stripped AiiDA `orm.Node` inputs down to lightweight `{pk, uuid, type}` stubs. To support detailed dictionary inspection in frontends without extra remote calls, structures such as `orm.Dict` explicitly extract `value.get_dict()` during transit.
