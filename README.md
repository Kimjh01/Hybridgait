python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main_evaluation.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0 --root <dir_containing_data> --gen_path <path_test_generated_data>


python -m torch.distributed.launch --nproc_per_node=1 --master_port 12345 main_evaluation.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0 --root data --gen_path ./disc/prcc_query


<<<<<<< HEAD:README.md
python main_evaluation.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0 --root data --gen_path ./disc/prcc_query 
=======
python main_evaluation.py --dataset prcc --cfg configs/res50_cels_cal.yaml --gpu 0 --root data --gen_path ./disc/prcc_query
>>>>>>> f3f4a80efcad49ecd053d3b5dfc2643e0d8a32e2:OpenGait-master/README.md

==========================================================================================

python datasets/GREW/rearrange_GREW.py --input_path D:\Grew_reduce --output_path D:\Grew_reduce_rearranged
python datasets/GREW/rearrange_GREW_pose.py --input_path D:\Grew_reduce --output_path D:\Grew_reduce_pose_rearranged
python datasets/pretreatment.py --input_path D:\Grew_reduce_rearranged --output_path D:\Grew_reduce_pkl --dataset GREW
python datasets/pretreatment.py --input_path D:\Grew_reduce_pose_rearranged --output_path D:\Grew_reduce_pose_pkl --pose --dataset GREW


python datasets/pretreatment_heatmap.py --pose_data_path D:\Grew_reduce_pose_pkl --save_root D:\Grew_reduce_posemap --dataset_name GREW
 -> pretreatment_heatmap  수정

python datasets/ln_sil_heatmap.py --heatmap_data_path D:\Grew_reduce_posemap\GREW_sigma_8.0_\pkl --silhouette_data_path D:\Grew_reduce_pkl --output_path D:\Grew_reduce_dataset
-> ln_sil_heatmap.py  수정

python opengait/main.py --cfgs configs\skeletongait\skeletongait++_GREW.yaml --phase train --log_to_file
->utils/msg_manager.py, utils/common.py 수정
-> data/sampler.py 수정
-> modeling/losses/base.py


