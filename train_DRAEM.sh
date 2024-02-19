python train_DRAEM.py \
 --gpu_id 0 \
 --obj_id 1 \
 --lr 0.0001 \
 --bs 1 \
 --epochs 700 \
 --data_path ../../MyData/anomaly_detection/MVTec \
 --anomaly_source_path ../../MyData/anomal_source/dtd_images \
 --checkpoint_path ./result/ \
 --log_path ./logs/