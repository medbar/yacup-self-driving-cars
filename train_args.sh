
set -eou pipefail
help_message="$0 config"


if [ $# -ne 1 ] ; then 
    echo "$help_message"
    exit 1
fi
config=$1

exp_dir=exp/$(dirname $config)/$(basename $config .yaml)
mkdir -p $exp_dir
inex \
        --log-level INFO \
        -s . \
        -u exp_dir=$exp_dir \
        -f $exp_dir/final_config.yaml \
        $config  2>&1 | tee $exp_dir/train.log 
