sudo pip3 install tensorflow==1.12.0
sudo pip3 install matplotlib
sudo pip3 install seaborn
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo
export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH