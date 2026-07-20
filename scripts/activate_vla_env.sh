#!/usr/bin/env bash

# Source this file so pip-installed CUDA libraries are visible to TensorFlow.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "Run: source scripts/activate_vla_env.sh" >&2
  exit 1
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
virtual_env="$project_root/.venv"
if [[ ! -f "$virtual_env/bin/activate" ]]; then
  echo "Missing $virtual_env; create it with: python3 -m venv .venv" >&2
  return 1
fi

# shellcheck disable=SC1091
source "$virtual_env/bin/activate"

site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
nvidia_packages="$site_packages/nvidia"
if [[ -d "$nvidia_packages" ]]; then
  cuda_library_path="$(
    find "$nvidia_packages" -type d \( -name lib -o -name lib64 \) -print \
      | sort \
      | paste -sd: -
  )"
  if [[ -n "$cuda_library_path" ]]; then
    export LD_LIBRARY_PATH="$cuda_library_path${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi

unset project_root virtual_env site_packages nvidia_packages cuda_library_path
