#!/usr/bin/env sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
output_dir="$project_dir/build/srm"
mkdir -p "$output_dir"

${CXX:-c++} -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter \
  "$project_dir/third_party/srm/portable_main.cpp" \
  "$project_dir/third_party/srm/SRM_src/SRMclass.cpp" \
  "$project_dir/third_party/srm/SRM_src/submodel.cpp" \
  "$project_dir/third_party/srm/SRM_src/s.cpp" \
  "$project_dir/third_party/srm/SRM_src/image.cpp" \
  -I"$project_dir/third_party/srm/SRM_src" \
  -o "$output_dir/srm_extract"

echo "$output_dir/srm_extract"
