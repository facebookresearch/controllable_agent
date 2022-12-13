#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

func () {

  # env name either specified as second arg, or defaults to ca
  local ENVNAME="${2:-ca}"  

  case $1 in
    install)
      _install "$ENVNAME"
      ;;
    activate)
      _activate "$ENVNAME"
      ;;
    *)
      echo "First parameter must be either install or activate"
      ;;
  esac
}


_install () {
  # installs the whole environment, including mujoco
  _install_mujoco
  _activate "$1"
  local CONDA_HOME=$HOME/.conda
  if [ -d "$CONDA_HOME" ]; then \
    conda install -n "$1" cudatoolkit=11.1 -c pytorch -y; \
    conda activate "$1"  # just to be sure
  fi
  which pip
  pip install  --progress-bar off -U pip wheel
  pip install  --progress-bar off -r requirements.txt
  # required? unclear
  # conda install -c anaconda patchelf
  # conda install glew
}


_activate () {
  # activates environment (conda env in priority)
  local CONDA_HOME=$HOME/.conda
  local CONDA_ENV=$CONDA_HOME/envs/$1
  if [ -d "$CONDA_HOME" ]; then
    echo "Activating conda environment:" "$CONDA_ENV"
    conda deactivate
    if [ ! -d "$CONDA_ENV" ]; then
      echo "Creating environment"
      conda create -n "$1" python=3.8 ipython -y
    fi
    conda activate "$1"
  else
    local VENV_PATH=$HOME/.venvs/$1
    echo "Activating venv environment:" "$VENV_PATH"
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH"/bin/activate
  fi
}


_install_mujoco () {
  # copy mujoco source to ~/.mujoco
  local MUJOCO_FOLDER=$HOME/.mujoco
  if [ ! -d "$MUJOCO_FOLDER" ]; then
    local MACHINE
    MACHINE="$(uname -s)"
    local MUJOCO_FILE=mujoco-release-tmp.tar.gz 
    mkdir "$MUJOCO_FOLDER"
    case $MACHINE in
      Darwin)
        echo "Installation of MuJoCo.framework in ~/.mujoco is unreliable on Mac, you may have to do it yourself"
        wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-macos-universal2.dmg -O $MUJOCO_FILE
        hdiutil attach $MUJOCO_FILE
        cp -r /Volumes/MuJoCo/MuJoCo.framework "$MUJOCO_FOLDER"
        hdiutil detach /Volumes/MuJoCo
        ;;
      Linux)
        wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz -O $MUJOCO_FILE
        # wget https://github.com/deepmind/mujoco/releases/download/2.3.0/mujoco-2.3.0-linux-x86_64.tar.gz -O $MUJOCO_FILE
        tar -zxvf $MUJOCO_FILE -C "$MUJOCO_FOLDER"
        ;;
      *)
        echo "Unsupported installation of Mujoco for machine" "$MACHINE"
        ;;
    esac
    rm -rf "$MUJOCO_FILE"
  fi
}


func "$1" "$2"  # (install | activate) (optional:env_name)
