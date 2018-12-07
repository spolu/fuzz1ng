# Fuzz1ng

### Run an experiment

EXPERIMENT=20181207_1222_checksum_8_8
mkdir -p ~/tmp/fuzzing/$EXPERIMENT
genetic_simple_fuzzer configs/dev.json ~/tmp/fuzzing/$EXPERIMENT/ --genetic_simple_sample_count=16 --gym_fuzz1ng_env=FuzzChecksum_8_8-v0
transformer_trainer configs/dev.json ~/tmp/fuzzing/$EXPERIMENT --transformer_save_dir=~/tmp/fuzzing/$EXPERIMENT --gym_fuzz1ng_env=FuzzChecksum_8_8-v0 --device=cuda:0 --genetic_simple_sample_count=16 --tensorboard_log_dir=~/tmp/tensorboard/$EXPERIMENT_`now`

### Evaluate a runs_db with afl-cmin

afl_dump configs/dev.json ~/tmp/fuzzing/20181128_1300_genetic/ ~/tmp/fuzzing/input_all
afl-cmin -i ~/tmp/fuzzing/input_all/ -o ~/tmp/fuzzing/input_cmin/ -- ~/opt/libpng_simple_fopen_afl @@
