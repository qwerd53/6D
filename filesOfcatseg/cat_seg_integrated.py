import os
import torch
import numpy as np
from PIL import Image
import cv2

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from detectron2.utils.file_io import PathManager

from cat_seg import add_cat_seg_config
from cat_seg.third_party import clip
from cat_seg.third_party import imagenet_templates


class CATSegIntegrated:
    def __init__(self, config_file, model_weights, device=None):
        """
        初始化CAT-Seg模型

        Args:
            config_file: 配置文件路径
            model_weights: 预训练模型权重路径
            device: 运行设备，默认为cuda（如果可用）
        """
        # 设置设备
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        # 加载配置
        self.cfg = self._setup_config(config_file)

        # 构建模型
        self.model = self._build_model(self.cfg)

        # 加载权重
        self._load_weights(self.model, model_weights)

        # 设置为评估模式
        self.model.eval()

        # 获取元数据
        self.metadata = MetadataCatalog.get(
            self.cfg.DATASETS.TEST[0] if len(self.cfg.DATASETS.TEST) else "__unused"
        )

    def _setup_config(self, config_file):
        """设置模型配置"""
        cfg = get_cfg()
        # 添加deeplab和cat_seg的配置
        from detectron2.projects.deeplab import add_deeplab_config
        add_deeplab_config(cfg)
        add_cat_seg_config(cfg)

        # 从文件加载配置
        cfg.merge_from_file(config_file)

        # 冻结配置
        cfg.freeze()

        return cfg

    def _build_model(self, cfg):
        """构建模型"""
        model = build_model(cfg)
        model.to(self.device)
        return model

    def _load_weights(self, model, weights_path):
        """加载模型权重"""
        checkpointer = DetectionCheckpointer(model, save_dir=self.cfg.OUTPUT_DIR)
        checkpointer.load(weights_path)

    def _preprocess_image(self, image):
        """预处理输入图像"""
        # 如果输入是文件路径，读取图像
        if isinstance(image, str):
            with PathManager.open(image, "rb") as f:
                image = Image.open(f).convert("RGB")
            image = np.array(image)

        # 如果输入是PIL Image，转换为numpy数组
        elif isinstance(image, Image.Image):
            image = np.array(image)

        # 如果图像是BGR格式（OpenCV读取的），转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 检查是否为BGR格式（简单判断：如果均值偏暗，可能是BGR）
            if image.mean() < 100 and image[..., 0].mean() > image[..., 2].mean():
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为张量并标准化
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

        # 应用CLIP的像素标准化
        pixel_mean = torch.tensor(self.cfg.MODEL.CLIP_PIXEL_MEAN).view(-1, 1, 1).to(self.device)
        pixel_std = torch.tensor(self.cfg.MODEL.CLIP_PIXEL_STD).view(-1, 1, 1).to(self.device)
        image_tensor = (image_tensor - pixel_mean) / pixel_std

        return image_tensor, image.shape[:2]  # 返回处理后的张量和原始图像尺寸

    def predict(self, image, prompt=None, custom_classes=None):
        """
        进行分割预测

        Args:
            image: 输入图像（可以是文件路径、PIL Image或numpy数组）
            prompt: 可选的文本提示
            custom_classes: 可选的自定义类别列表

        Returns:
            logits: 分割掩码的logits (C, H, W)
        """
        with torch.no_grad():
            # 预处理图像
            image_tensor, original_size = self._preprocess_image(image)

            # 构建输入批次
            inputs = [{
                "image": image_tensor,
                "height": original_size[0],
                "width": original_size[1]
            }]

            # 执行推理
            outputs = self.model(inputs)

            # 提取分割掩码logits
            sem_seg = outputs[0]["sem_seg"]

            return sem_seg

    def predict_with_prompt(self, image, prompt, template="A photo of a {} in the scene"):
        """
        使用自定义提示进行分割预测

        Args:
            image: 输入图像（可以是文件路径、PIL Image或numpy数组）
            prompt: 文本提示，例如"person"、"car"等
            template: 提示模板，用于格式化提示

        Returns:
            logits: 分割掩码的logits (C, H, W)
        """
        with torch.no_grad():
            # 预处理图像
            image_tensor, original_size = self._preprocess_image(image)

            # 构建输入批次
            inputs = [{
                "image": image_tensor,
                "height": original_size[0],
                "width": original_size[1]
            }]

            # 获取模型中的clip_model
            clip_model = self.model.sem_seg_head.predictor.clip_model

            # 根据模板格式化提示
            formatted_prompt = template.format(prompt)

            # 标记化提示文本
            tokenizer = self.model.sem_seg_head.predictor.tokenizer
            if tokenizer is not None:
                # OpenCLIP模型的处理方式
                text_tokens = tokenizer([formatted_prompt]).to(self.device)
            else:
                # OpenAI CLIP模型的处理方式
                text_tokens = clip.tokenize([formatted_prompt]).to(self.device)

            # 编码文本特征
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # 保存原始的text_features和测试类别
            original_text_features = self.model.sem_seg_head.predictor.text_features_test.clone()
            original_test_classes = self.model.sem_seg_head.predictor.test_class_texts

            try:
                # 临时替换为自定义提示的特征
                self.model.sem_seg_head.predictor.text_features_test = text_features.unsqueeze(0).permute(1, 0,
                                                                                                          2).float()
                self.model.sem_seg_head.predictor.test_class_texts = [prompt]

                # 执行推理
                outputs = self.model(inputs)

                # 提取分割掩码logits
                sem_seg = outputs[0]["sem_seg"]

                return sem_seg
            finally:
                # 恢复原始的text_features和测试类别
                self.model.sem_seg_head.predictor.text_features_test = original_text_features
                self.model.sem_seg_head.predictor.test_class_texts = original_test_classes


# 使用示例
if __name__ == "__main__":
    # 配置和权重路径
    config_path = "configs/config.yaml"
    weights_path = "path/to/your/model_weights.pth"

    # 初始化模型
    cat_seg = CATSegIntegrated(config_path, weights_path)

    # 示例1：使用图像路径
    image_path = "path/to/your/image.jpg"
    logits = cat_seg.predict(image_path)
    print(f"预测结果形状: {logits.shape}")

    # 示例2：使用自定义提示
    prompt_logits = cat_seg.predict_with_prompt(image_path, "person", template="A photo of a {} in the scene")
    print(f"使用提示的预测结果形状: {prompt_logits.shape}")