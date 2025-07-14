# TEM-Seg Implementation TODOs

This document tracks tasks needed to align our current implementation with the original implementation strategy and scaffold.

## Structural Refinements

### Scale Estimation System
- [ ] Refactor `scale_estimation.py` into a proper module structure:
  ```
  scale_estimation/
  ├── __init__.py
  ├── estimators.py       # Move ScaleEstimator and implementations here
  ├── manager.py          # Move ScaleManager here
  └── utils.py            # Move utility methods here
  ```
- [ ] Update imports and references throughout codebase
- [ ] Ensure tests remain functional after refactoring

### CLI Structure
- [ ] Decide whether to:
  - [ ] Rename `cmd/` directory to `cli/` to match the documentation, OR
  - [ ] Update documentation to reflect current `cmd/` naming
- [ ] Ensure consistent naming convention across all documentation

### Module Organization
- [ ] Review and refine module organization to align with the planned structure
- [ ] Consider more granular separation of concerns in larger files
- [ ] Ensure proper docstrings and type hints throughout

## Implementation Schedule

### Current Status
- [x] Scale Estimation System
  - [x] Core implementation
  - [x] CLI integration
  - [x] Comprehensive testing
  - [x] Debug mode and error reporting
  - [x] Documentation

### Next Phase
- [ ] Configuration System Foundation
  - [ ] Define configuration dataclasses
  - [ ] Implement YAML serialization/deserialization
  - [ ] Create configuration validation logic
  - [ ] Implement hash-based freezing mechanism
  - [ ] Add organelle-specific presets

### Future Work
- [ ] Workflow Infrastructure
  - [ ] SegmentationWorkflow abstract base
  - [ ] StandardizedOutput implementation
  - [ ] Workflow directory management
  - [ ] Workflow Registry and Factory

## Technical Debt and Improvements

- [ ] Consider optimizations for scale detection
  - [ ] Caching of scale estimates for performance
  - [ ] Additional scale estimation strategies
  - [ ] Performance profiling of scale bar detection
- [ ] Legacy code modernization
  - [ ] Further refinements to calibration code
  - [ ] Complete migration away from deprecated NumPy functions
