# Changelog

All notable changes to the Multiplexed Holographic Metasurfaces project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive main README.md with project overview and usage examples
- CONTRIBUTING.md with development guidelines
- requirements.txt for Python dependencies
- CHANGELOG.md for tracking project changes
- Scripts directory with README for legacy code organization

### Changed
- Reorganized repository structure
  - Moved legacy Python scripts to `scripts/legacy/`
  - Moved MATLAB pipeline to `scripts/legacy/`
  - Moved MAT data files to `data/raw/`
  - Consolidated target images in `data/targets/common/`
- Enhanced .gitignore with comprehensive patterns
- Improved documentation consistency across modules

### Removed
- desktop.ini (Windows-specific file)
- config file from root directory
- Redundant targets directory (consolidated into data/targets/)

## [0.2.0] - Prior Refactoring

### Added
- Modular source code structure under `src/`
- CLI tools for metasurface library processing
- Comprehensive module-level READMEs
- Jupyter notebook tutorials
- Automated output organization with metadata

### Changed
- Refactored notebooks to use src modules
- Standardized CLI patterns
- Improved output directory organization

## [0.1.0] - Initial Development

### Added
- GS+ASM hologram generation algorithm
- Dammann grating generation
- Metasurface library processing tools
- Basic optimization algorithms
- Legacy phase matching code

---

## Version Guidelines

- **Major version** (X.0.0): Breaking changes, major restructuring
- **Minor version** (0.X.0): New features, backward-compatible changes
- **Patch version** (0.0.X): Bug fixes, documentation updates

## Categories

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security-related changes
