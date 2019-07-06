################################################################################
# Setup script for maopy
################################################################################
# PATHS
################################################################################
ROOT=$(pwd)
BUILD=$ROOT/build
THIRD_PARTY=$ROOT/third_party
PATH=$ROOT/build/bin:$PATH

################################################################################
# Build Config
################################################################################
AUTOMAKE_JOBS=$(nproc)

################################################################################
# Package Versions
################################################################################
OMPI_VERSION="v3.1.4"
M4_VERSION="1.4.17"
AUTOCONF_VERSION="2.69"
AUTOMAKE_VERSION="1.15"
LIBTOOL_VERSION="2.4.6"

CONDA_VERSION="4.6.14"
PYTHON_VERSION="3.6"

################################################################################
# Instalation Functions
################################################################################
function install_GNU_package () {
    package_name=$1
    version=$2

    GNU_FTP=ftp://ftp.gnu.org/gnu
    package=$package_name-$version

    cd $THIRD_PARTY

    if [ ! -d $package ]; then
        tar=$package.tar.gz

        wget $GNU_FTP/$package_name/$tar
        tar -xf $tar
        rm $tar

        cd $package
        ./configure --prefix=$BUILD
        make
        make install
    fi
}

################################################################################
# Main
################################################################################

# Install Open MPI dependencies
install_GNU_package m4 $M4_VERSION
install_GNU_package autoconf $AUTOCONF_VERSION
install_GNU_package automake $AUTOMAKE_VERSION
install_GNU_package libtool $LIBTOOL_VERSION

# Install Open MPI
cd $THIRD_PARTY/ompi
git checkout $OMPI_VERSION

perl autogen.pl -d
./configure --prefix=$BUILD
make
make install

# Create Conda Enviroment
system_conda_version=$(conda --version  | cut -d' ' -f2)
oldest_conda_version=$(echo -e "$CONDA_VERSION\n$system_conda_version" \
                       | sort -V | head -1)

if [ $oldest_conda_version != $CONDA_VERSION ]; then
    echo 'Conda must be at least version $CONDA_VERSION'
    exit
fi

enviroment_name="maopy"

if [ ! -z $(conda env list | grep $enviroment_name) ]; then
    echo "Envoriment $enviroment_name alreay exists"
    #exit
fi

yes | conda create -n $enviroment_name python=$PYTHON_VERSION
eval "$(conda shell.bash hook)" #TODO: This is needed to allow virtual envs
conda activate $enviroment_name

# Install mpi4py
yes | conda install cython numpy
cd $THIRD_PARTY/mpi4py
python setup.py install
