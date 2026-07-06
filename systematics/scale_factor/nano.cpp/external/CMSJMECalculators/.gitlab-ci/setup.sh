#!/usr/bin/env bash

action() {
    local shell_is_zsh="$( [ -z "${ZSH_VERSION}" ] && echo "false" || echo "true" )"
    local this_file="$( ${shell_is_zsh} && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "${this_file}" )" && pwd )"

    #
    # CMSJMECalculators setup
    #

    #
    # Environment
    #

    # Get micromamba
    export HOME="${this_dir}/install"  # Don't pollute the user's home folder
    mkdir -p "$HOME"
    if ! [ -d "$HOME/micromamba" ] ; then
        echo | bash <(curl -L micro.mamba.pm/install.sh)
    fi

    # Activate the environment
    source "$HOME/.bashrc"
    micromamba activate

    # Make sure build dependencies are installed
    local packages
    function check_package() {
        which $1 &>/dev/null || packages="$packages $2"
    }
    check_package root       root
    check_package g++        gxx
    check_package cmake      cmake
    check_package correction correctionlib
    check_package pytest     pytest
    check_package boost      boost

    if [ -n "$packages" ] ; then
        echo "Installing: $packages"
        micromamba install -y $packages
    fi
}
action "$@"
