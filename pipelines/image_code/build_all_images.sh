cd 00_metadata_logger
. ./build_image.sh
cd ..
echo "00_metadata_logger built"
cd 01_prepare_data
. ./build_image.sh
cd ..
echo "01_prepare_data built"
cd 02_prepare_embeddings
. ./build_image.sh
cd ..
echo "02_prepare_embeddings built"
cd 03_generate_model
. ./build_image.sh
cd ..
echo "03_generate_model built"
cd 04_move_data_pvc
. ./build_image.sh
cd ..
echo "04_move_data_pvc built"
cd 05_train_model
. ./build_image.sh
. /build_image_gpu.sh
cd ..
echo "05_train_model built"
cd 06_dist_launcher
. ./build_image.sh
cd ..
echo "06_dist_launcher built"
cd 07_deploy_model
. ./build_image.sh
cd ..
echo "07_deploy_model built"