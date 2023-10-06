FROM python:3.10-slim

RUN apt-get update                                                                          \
    && apt-get install -y git                                                               \
    && export DEBIAN_FRONTEND=noninteractive                                                \
    && apt-get update -y                                                                    \
    && apt-get -y install tmux                                                              \
    && apt-get install unzip -y                                                             \
    && apt-get install software-properties-common -y                                        \
############
            build-essential python3-dev curl wget                                \
            libssl-dev                                                                      \
            libffi-dev                                                                      \
            libkrb5-dev                                                                     \
            liblzma-dev                                                                     \
            jq                                                                              \
            vim
############

#install python pip in the image
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# install python specific packages
COPY requirements.txt .
RUN pip3 install --user -r requirements.txt

# install zsh
RUN apt install zsh -y
RUN sh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"
