
set -eou pipefail
venv_dir=anaconda

if [ ! -f $venv_dir/.done ]; then                                                      
    echo "$0: Download anaconda"                                                  
    installer=Miniconda3-py39_23.11.0-2-Linux-x86_64.sh                                
    [ ! -f $installer ] && \
        wget --retry-connrefused --waitretry=1 --read-timeout=20 --timeout=15 -t 20 \
            https://repo.anaconda.com/miniconda/$installer                             
    bash $installer -b -p $venv_dir                                                    
    touch $venv_dir/.done                                                              
fi                                                                                         

source $venv_dir/bin/activate
conda create -y --name yacup python=3.9
conda activate yacup
conda install -y pip 
pip install numpy pandas scipy scikit-learn ipython jupyter traceback-with-variables
pip install matplotlib seaborn
pip install lightning 
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/medbar/webdataset.git


pip install -U inex-launcher
