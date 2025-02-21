#!/bin/bash

function print_the_help {
  echo "USAGE: ${0} -- <install_directory>"
  echo "  OPTIONS: "
  echo "    --install_directory     where fastjet and fastjet-contrib will be installed"
  echo "    -h, --help                print this help menu"
}

install_directory="./"
original_directory=$PWD
exit_script=false
while true; do
    case "$1" in
        --help|-h)
            print_the_help
            exit_script=true
            break
            ;;
        --install_directory)
            if [ -n "$2" ]; then
                install_directory="$2"
                shift 2
            else
                echo "Error: --install_directory requires an argument."
                print_the_help
                exit_script=true
                break
            fi
            ;;
        "")
            break
            ;;
        *)
            echo "Error: Invalid option '$1'."
            print_the_help
            exit_script=true
            break
            ;;
    esac
done
if [ "$exit_script" = true ]; then
    if [ "$0" = "$BASH_SOURCE" ]; then
        exit
    else
        return
    fi
fi

echo "Install directory set to ${install_directory}"

# Installing fastjet
cd $install_directory
curl -O https://fastjet.fr/repo/fastjet-3.4.3.tar.gz
tar zxvf fastjet-3.4.3.tar.gz
cd fastjet-3.4.3/
./configure --prefix=$PWD/../fastjet-3.4.3-install
make -j20
make install -j20
cd ..

# Installing fjcontrib
wget http://fastjet.hepforge.org/contrib/downloads/fjcontrib-1.100.tar.gz
tar xzvf fjcontrib-1.100.tar.gz
cd fjcontrib-1.100
./configure --fastjet-config=$PWD/../fastjet-3.4.3-install/bin/fastjet-config
make -j20
make install -j20
cd ..
rm fjcontrib-1.100.tar.gz
rm fastjet-3.4.3.tar.gz
cd ${original_directory}

echo "Finished installation"