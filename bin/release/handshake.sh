#!/bin/bash
# Linux equivalent of handshake.bat
# Performs OpenCL database handshaking and distributes KernelTimeSpecification.list

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Step 1: Copy primitive.cl from Handshaking to current directory
cp ./Handshaking/primitive.cl ./primitive.cl

# Step 2: Run the handshake executable
./Handshaking/DLL_OpenCL

# Step 3: Distribute KernelTimeSpecification.list to scheduler directories
cp KernelTimeSpecification.list ./KernelScheduler/KernelTimeSpecification.list
cp KernelTimeSpecification.list ./OperatorSheduler/KernelTimeSpecification.list
cp KernelTimeSpecification.list ./QueryScheduler/KernelTimeSpecification.list

# Step 4: Clean up - remove temporary primitive.cl and move list back to Handshaking
rm -f primitive.cl
mv KernelTimeSpecification.list ./Handshaking/KernelTimeSpecification.list

echo "Handshake complete."
