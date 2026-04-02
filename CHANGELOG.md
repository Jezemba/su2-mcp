# Changelog

## Unreleased

### Added
- `get_valid_config_options` tool: returns the ~30 most common SU2 v8.3
  Euler/RANS config options with types and descriptions, plus a mapping of
  deprecated v5/v6 option names.
- Automatic deprecated-option remapping in `update_config_entries`: when a
  caller uses a deprecated SU2 v5/v6 option name (e.g. `PHYSICAL_PROBLEM`),
  it is silently translated to the SU2 v8.3 equivalent (`SOLVER`) and a
  warning is returned so the agent learns the correct name.
