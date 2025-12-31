# CHANGELOG

## 0.2.1 (in development)
### Features
- None

### Refactor
- **Simplified Agent API**: Consolidated all agent functionality under `Agent` class for better consistency
  - Moved `load_from_yaml()` from `AgentManager` to `Agent` as a classmethod
  - Users now use `Agent.load_from_yaml()` instead of `AgentManager.load_from_yaml()`
  - Removed `AgentManager` class entirely
- **Removed `system_message` field**: Now automatically computed from `role` and `instructions` instead of being a configurable field, reducing redundancy
- **Removed unused methods**: `to_dict()` and `from_dict()` from `Agent` class (not used anywhere in codebase)
- **Code organization**: Improved inline class documentation and structure

### Bug Fix
- None

### Breaking Changes and Migration Guide
- **Removed `AgentManager` class**
  - **Migration**: Replace `AgentManager.load_from_yaml(path)` with `Agent.load_from_yaml(path)`
  - **Migration**: If you were using `AgentManager()` for collection management, use a simple `Dict[str, Agent]` instead
  
- **Removed `system_message` parameter**
  - **Migration**: Remove `system_message` from YAML configs or `Agent` constructor calls
  - The system message is now automatically generated as `f"You are a {role}. {instructions}"`

### Documentation
- Updated README.md to reflect simplified API structure
  - Removed `AgentManager` API reference section
  - Updated Quick Start examples to use `Agent.load_from_yaml()`
  - Updated Agent Configuration section with clearer guidance (YAML recommended for multi-agent workflows)
  - Removed internal methods (`get_tools()`, `get_llm_client()`) from public API reference to minimize API surface
- Updated all example files to use `Agent.load_from_yaml()` instead of `AgentManager.load_from_yaml()`

## 0.2.0 (2025-12-23)
- Alpha release

## 0.1.x 
- Pre-alpha release