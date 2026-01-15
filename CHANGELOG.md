# CHANGELOG

## 0.2.1 (in development)
### Features
- **YAML-Only Agent Configuration**: Agents must now be configured via YAML files. The `Agent(file_path)` constructor automatically loads from YAML, supporting both single and multiple agents.

### Refactor
- **Simplified Agent API**: 
  - `Agent(file_path)` now automatically loads from YAML - returns single `Agent` for one agent, or `List[Agent]` for multiple agents
  - Removed `load_from_yaml()` from public API (now private `_load_from_yaml()`)
  - Removed automatic path resolution - users handle paths themselves using `os.path.join()`
  - Removed support for dict format in YAML - all YAML files must be lists
- **Updated Examples**: All examples now use `Agent(config_path)` instead of `Agent.load_from_yaml(config_path)`
- **Renamed Config Files**: All YAML config files now have number prefixes matching their example files (e.g., `01_basic_simple.yaml`)
- **Consolidated Agent functionality**: Moved `load_from_yaml()` from `AgentManager` to `Agent` class, removed `AgentManager` class entirely
- **Removed `system_message` field**: Now automatically computed from `role` and `instructions` instead of being a configurable field, reducing redundancy
- **Removed unused methods**: `to_dict()` and `from_dict()` from `Agent` class (not used anywhere in codebase)
- **Code organization**: Improved inline class documentation and structure
- **Method inlining**: Inlined some single-use methods to reduce indirection and improve code clarity

### Bug Fix
- None

### Breaking Changes and Migration Guide
- **YAML-Only Configuration Required**
  - **Migration**: All agents must now be configured via YAML files. Direct constructor with keyword arguments is no longer supported.
  - **Before**: `Agent(name="agent", role="Role", instructions="...")`
  - **After**: Create a YAML file and use `Agent("configs/agent.yaml")`
  
- **Removed `Agent.load_from_yaml()` from public API**
  - **Migration**: Use `Agent(config_path)` instead. For single agents, it returns an `Agent` instance. For multiple agents, it returns a `List[Agent]`.
  - **Before**: `agents = Agent.load_from_yaml("configs/agents.yaml")`
  - **After**: `agents = Agent("configs/agents.yaml")` (works for both single and multiple agents)

- **YAML Format Change**
  - **Migration**: YAML files must now always be lists (even for single agents). Remove any dict-format YAML configs.
  - **Before**: `name: agent1` (dict format)
  - **After**: `- name: agent1` (list format)
  
- **Removed `AgentManager` class** (from previous version)
  - **Migration**: Replace `AgentManager.load_from_yaml(path)` with `Agent(path)`
  - **Migration**: If you were using `AgentManager()` for collection management, use a simple `Dict[str, Agent]` instead
  
- **Removed `system_message` parameter** (from previous version)
  - **Migration**: Remove `system_message` from YAML configs or `Agent` constructor calls
  - The system message is now automatically generated as `f"You are a {role}. {instructions}"`

### Documentation
- Updated README.md to reflect simplified API structure
  - Removed `load_from_yaml()` from API reference (now private)
  - Updated all examples to use `Agent(config_path)` pattern
  - Updated Agent Configuration section to show YAML-only configuration
  - Updated path handling guidance
- Updated all example files to use `Agent(config_path)` instead of `Agent.load_from_yaml(config_path)`

## 0.2.0 (2025-12-23)
- Alpha release

## 0.1.x 
- Pre-alpha release