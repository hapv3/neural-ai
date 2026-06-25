#!/bin/bash
# Script to checkout correct versions of git submodules

echo "Initializing and updating submodules..."
git submodule sync --recursive
git submodule update --init --recursive

echo "Checking out specific versions noted in snitch_minimal.F..."
# hw/common_cells  (pulp-platform/common_cells @ snitch branch)
if [ -d "hw/common_cells" ]; then
    echo "Checking out hw/common_cells -> snitch branch"
    cd hw/common_cells
    git fetch origin
    git checkout snitch
    cd ../..
fi

# hw/fpnew         (pulp-platform/cvfpu        @ pulp-v0.2.3)
if [ -d "hw/fpnew" ]; then
    echo "Checking out hw/fpnew -> pulp-v0.2.3"
    cd hw/fpnew
    git fetch origin
    git checkout pulp-v0.2.3
    cd ../..
fi

# hw/tech_cells    (pulp-platform/tech_cells_generic @ v0.2.13)
if [ -d "hw/tech_cells" ]; then
    echo "Checking out hw/tech_cells -> v0.2.13"
    cd hw/tech_cells
    git fetch origin
    git checkout v0.2.13
    cd ../..
fi

# hw/riscv-dbg     (pulp-platform/riscv-dbg    @ v0.8.1)
if [ -d "hw/riscv-dbg" ]; then
    echo "Checking out hw/riscv-dbg -> v0.8.1"
    cd hw/riscv-dbg
    git fetch origin
    git checkout v0.8.1
    cd ../..
fi

echo "Submodule checkout complete. Current status:"
git submodule status
