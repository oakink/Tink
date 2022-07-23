# Tink

This is the official implementation of Tink.










## Step 6. 编译DeepSDF

参照 `DeepSDF/README.md`，注意，此处相对于原版DeepSDF有改动，一定要重新编译。

## Step 7. 预处理SDF

```shell
export MESA_GL_VERSION_OVERRIDE=3.3
export PANGOLIN_WINDOW_URI=headless://

cd DeepSDF

python preprocess_data.py --data_dir data/sdf/xxx --skip --threads 16

# 在data/sdf/xxx中，sdf文件存储在 SdfSamples下，物体重新align+rescale模型存放在 SdfSamples_resize下，rescale.pkl为该类物体统一scale的系数
```

## Step 8. 训练模型

```shell
cd DeepSDF

python train_deep_sdf.py -e data/sdf/xxx

# 网络模型存放在 data/sdf/xxx/network下
```

## Step 9. 导出 LatentCode

```shell
cd DeepSDF

python reconstruct_train.py -e data/sdf/xxx  # --mesh_include

# latentCode放在 data/sdf/xxx/Reconstructions/Codes下面
```

---

---

---

## Step 10. 插值

```shell
# 在yoda_object文件夹下
python gen_interpolate.py --all -d ./DeepSDF/data/sdf/xxx

# 插值的mesh在 DeepSDF/data/sdf/xxx/interpolate下
```

## Step 11. 计算contact info

```shell
# obj_id：grasp交互的物体id；tag: 批处理tag 详见yoda_hand；pose_path：手的pose shape tsl 文件路径
python yoda_object/cal_contact_info.py -d ./DeepSDF/data/sdf/xxx -s {obj_id} -t {tag} -p {pose_path}

# contact_info.pkl存储在 ./DeepSDF/data/sdf/xxx/contact/{obj_id}/{tag}_~~~~下。同时保存了物体坐标系下的hand pose参数和原pose路径
```

## Step 12. transfer contact info

```shell
# obj_id：grasp交互的物体id；tag: 批处理tag 详见yoda_hand；target_obj_id：迁移目标物体的id
python yoda_object/info_transform.py -d ./DeepSDF/data/sdf/xxx -p ./DeepSDF/data/sdf/xxx/contact/{obj_id}/{tag}_~~~~ -s {obj_id} -t {target_obj_id}

# contact_info.pkl存储在 ./DeepSDF/data/sdf/xxx/contact/{obj_id}/{tag}_~~~~/{target_obj_id}/下，同时保存了每一步迁移的contact info
```

## Step 13. fitting

```shell
# obj_id：grasp交互的物体id；tag: 批处理tag 详见yoda_hand；target_obj_id：迁移目标物体的id
# fix_tsl：将手的初值，从原contact region的质心，平移到新contact region的质心。实验表明会收敛更快且更稳定
CUDA_VISIBLE_DEVICES=0 python yoda_object/manip_gen.py -d ./DeepSDF/data/sdf/xxx -p ./DeepSDF/data/sdf/xxx/contact/{obj_id}/{tag}_~~~~ -s {obj_id} -t {target_obj_id} # --fix_tsl

# hand_param.pkl存储在 ./DeepSDF/data/sdf/xxx/contact/{obj_id}/{tag}_~~~~/{target_obj_id}/下，hand pose在物体坐标系中
```

## Step 14. 批处理脚本

批处理 **Step 11-13**：

```shell
# 详情参考代码逻辑
for i in {0..k - 1}; do python yoda_object/pipeline.py -d DeepSDF/data/sdf/xxx --tag {tag} --stage sss --wid $i --workers k --filter_dir {path to dom_filter}
```


