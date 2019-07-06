ROOT=$(pwd)
export PATH=$ROOT/build/bin:$PATH

PROCESSES=$1
ALGORITHM=$2

mpiexec -np $PROCESSES \
        --machinefile $ROOT/scripts/nodes.txt \
        python -m maopy.$(ALGORITHM)
