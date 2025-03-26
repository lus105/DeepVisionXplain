@echo off

:: Set environment variables
set pcb_data_path = %oc.env:pcb_data_path%
set gears_data_path = %oc.env:gears_data_path%

:: CNN
python src/eval.py experiment=eval_seg_xai model=cnn_effnet_v2_s_full data.data_dir=%pcb_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--efficientnet_v2_s_full_pcb
python src/eval.py experiment=eval_seg_xai model=cnn_effnet_v2_s_full data.data_dir=%gears_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--efficientnet_v2_s_full_gears

python src/eval.py experiment=eval_seg_xai model=cnn_effnet_v2_s_down data.data_dir=%pcb_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--efficientnet_v2_s_downscaled_pcb
python src/eval.py experiment=eval_seg_xai model=cnn_effnet_v2_s_down data.data_dir=%gears_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--efficientnet_v2_s_downscaled_gears

python src/eval.py experiment=eval_seg_xai model=cnn_mobnet_v3_large_full data.data_dir=%pcb_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--mobilenet_v3_large_full_pcb
python src/eval.py experiment=eval_seg_xai model=cnn_mobnet_v3_large_full data.data_dir=%gears_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--mobilenet_v3_large_full_gears

python src/eval.py experiment=eval_seg_xai model=cnn_mobnet_v3_large_down data.data_dir=%pcb_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--mobilenet_v3_large_downscaled_pcb
python src/eval.py experiment=eval_seg_xai model=cnn_mobnet_v3_large_down data.data_dir=%gears_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--mobilenet_v3_large_downscaled_gears

:: VIT
python src/eval.py experiment=eval_seg_xai model=vit_tiny data.data_dir=%pcb_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--vit_tiny_patch16_224.augreg_in21k_ft_in1k_pcb
python src/eval.py experiment=eval_seg_xai model=vit_tiny data.data_dir=%gears_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--vit_tiny_patch16_224.augreg_in21k_ft_in1k_gears

python src/eval.py experiment=eval_seg_xai model=vit_deit_tiny data.data_dir=%pcb_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--deit_tiny_patch16_224.fb_in1k_pcb
python src/eval.py experiment=eval_seg_xai model=vit_deit_tiny data.data_dir=%gears_data_path% model.ckpt_path=${paths.trained_models}/models--DeepVisionXplain--deit_tiny_patch16_224.fb_in1k_gears