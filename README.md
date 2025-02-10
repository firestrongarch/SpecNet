# SpecNet

## 说明
- 本项目基于 GAN 模型进行图像生成，包含训练和推断两个部分。
- 使用 predict.py 可加载预训练模型进行图像生成，请根据实际需求调整输入和输出路径。
- 在训练阶段，使用 GAN.py 中定义的 GAN 模型，依赖数据集需放置在指定路径下。
- 确保已安装所有依赖，推荐使用 requirements.txt 进行安装:
    ```sh
    pip3 install torch torchvision torchaudio -i https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    ```
