CUDA_VISIBLE_DEVICES=0 python examples/I2VGenXL/infer_i2vgen-xl.py --val_s 0 --val_e 10 &
CUDA_VISIBLE_DEVICES=1 python examples/I2VGenXL/infer_i2vgen-xl.py --val_s 10 --val_e 20 &
CUDA_VISIBLE_DEVICES=2 python examples/I2VGenXL/infer_i2vgen-xl.py --val_s 20 --val_e 30 &
CUDA_VISIBLE_DEVICES=3 python examples/I2VGenXL/infer_i2vgen-xl.py --val_s 30 --val_e 40 &
CUDA_VISIBLE_DEVICES=4 python examples/I2VGenXL/infer_i2vgen-xl.py --val_s 40 --val_e 50 &
CUDA_VISIBLE_DEVICES=5 python examples/I2VGenXL/infer_i2vgen-xl.py --val_s 50 --val_e 60 &
CUDA_VISIBLE_DEVICES=6 python examples/I2VGenXL/infer_i2vgen-xl.py --val_s 60 --val_e 70 &
CUDA_VISIBLE_DEVICES=7 python examples/I2VGenXL/infer_i2vgen-xl.py --val_s 70 --val_e 80 &