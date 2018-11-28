# Fuzz1ng

Experiments with various fuzzing strategies.


## Notes

### Evaluate a runs_db with afl-cmin

afl_dump configs/dev.json ~/tmp/fuzzing/20181128_1300_genetic/ ~/tmp/fuzzing/input_all
afl-cmin -i ~/tmp/fuzzing/input_all/ -o ~/tmp/fuzzing/input_cmin/ -- ~/opt/libpng_simple_fopen_afl @@
