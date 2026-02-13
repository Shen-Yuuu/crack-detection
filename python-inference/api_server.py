"""
裂纹检测 Python 推理服务
FastAPI REST API 包装层

提供 REST API 接口供 Java 后端调用
"""

import asyncio
import base64
import io
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 导入模型
from models.convnext_upernet import ConvNeXtUPerNet
from inference.tta_inference import EnhancedTTAInference

app = FastAPI(
    title="裂纹检测推理服务",
    description="基于 ConvNeXt-UPerNet 的道路裂纹检测 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量
model: Optional[ConvNeXtUPerNet] = None
tta_inference: Optional[EnhancedTTAInference] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    version: str


class DetectionResponse(BaseModel):
    success: bool
    message: str
    mask_base64: Optional[str] = None
    overlay_base64: Optional[str] = None
    confidence: Optional[float] = None
    crack_count: Optional[int] = None
    total_area: Optional[float] = None
    processing_time: Optional[float] = None


def load_model(model_path: str = None):
    """加载模型"""
    global model, tta_inference
    
    if model_path is None:
        # 默认模型路径 - 优先查找 outputs/best.pth
        default_paths = [
            "outputs/best.pth",
            "outputs/best_model.pth",
            "checkpoints/best.pth",
            "checkpoints/best_model.pth",
        ]
        model_path = os.environ.get("MODEL_PATH", None)
        
        if model_path is None:
            for path in default_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        if model_path is None:
            model_path = "outputs/best.pth"  # 使用默认路径
    
    if not os.path.exists(model_path):
        print(f"警告: 模型文件不存在: {model_path}")
        print("使用随机初始化模型（仅用于测试）")
        model = ConvNeXtUPerNet(
            num_classes=1,
            encoder_name='convnext_small',
            pretrained=False
        )
    else:
        print(f"加载模型: {model_path}")
        model = ConvNeXtUPerNet(
            num_classes=1,
            encoder_name='convnext_small',
            pretrained=False
        )
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    # 创建 TTA 推理器
    tta_inference = EnhancedTTAInference(
        scales=[0.75, 1.0, 1.25],
        flip_h=True,
        flip_v=True,
        rotate_90=True,
        weight_by_scale=True,
        use_edge_aware=True
    )
    
    print(f"模型加载完成，使用设备: {device}")


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    load_model()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        device=str(device),
        version="1.0.0"
    )


@app.get("/api/v1/inference/health", response_model=HealthResponse)
async def inference_health():
    """推理服务健康检查（兼容路由）"""
    return await health_check()


@app.post("/api/v1/inference/detect", response_model=DetectionResponse)
async def detect_crack(
    file: UploadFile = File(None, alias="file", description="待检测的图像文件"),
    image: UploadFile = File(None, alias="image", description="待检测的图像文件（别名）"),
    threshold: float = Form(default=0.5, description="检测阈值"),
    use_tta: bool = Form(default=True, description="是否使用 TTA"),
    return_mask: bool = Form(default=True, description="是否返回mask"),
    return_overlay: bool = Form(default=True, description="是否返回叠加图像")
):
    """
    检测单张图像中的裂纹
    
    - **file/image**: 上传的图像文件（支持 jpg, png 等格式）
    - **threshold**: 检测阈值，默认0.5
    - **use_tta**: 是否使用测试时增强（TTA），默认开启
    - **return_mask**: 是否返回mask图像
    - **return_overlay**: 是否返回原图叠加检测结果的图像
    """
    # 兼容 file 和 image 两种参数名
    upload_file = file if file is not None else image
    if upload_file is None:
        raise HTTPException(status_code=400, detail="请上传图像文件")
    
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        start_time = time.time()
        
        # 读取图像
        contents = await upload_file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图像文件")
        
        # 转换为 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 预处理
        img_tensor = preprocess_image(img_rgb)
        
        # 推理
        with torch.no_grad():
            if use_tta and tta_inference is not None:
                # EnhancedTTAInference expects (C, H, W)
                pred = tta_inference(model, img_tensor.squeeze(0), device=str(device))
            else:
                outputs = model(img_tensor)
                pred = outputs['out'] if isinstance(outputs, dict) else outputs
                pred = torch.sigmoid(pred)
        
        # 后处理 - 使用 threshold 参数
        mask = pred.squeeze().cpu().numpy()
        mask_binary = (mask > threshold).astype(np.uint8) * 255
        
        # 调整回原始尺寸
        mask_resized = cv2.resize(mask_binary, (img.shape[1], img.shape[0]))
        
        # 计算统计信息
        confidence = float(mask.max())
        crack_count, total_area = calculate_statistics(mask_resized)
        
        # 编码 mask 为 base64 (根据 return_mask 参数)
        mask_base64 = None
        if return_mask:
            _, mask_encoded = cv2.imencode('.png', mask_resized)
            mask_base64 = base64.b64encode(mask_encoded.tobytes()).decode('utf-8')
        
        # 可选：生成叠加图像
        overlay_base64 = None
        if return_overlay:
            overlay = create_overlay(img, mask_resized)
            _, overlay_encoded = cv2.imencode('.png', overlay)
            overlay_base64 = base64.b64encode(overlay_encoded.tobytes()).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            success=True,
            message="检测成功",
            mask_base64=mask_base64,
            overlay_base64=overlay_base64,
            confidence=confidence,
            crack_count=crack_count,
            total_area=total_area,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.post("/api/v1/inference/detect-base64", response_model=DetectionResponse)
async def detect_crack_base64(
    image_base64: str = Form(..., description="Base64 编码的图像"),
    use_tta: bool = Form(default=True, description="是否使用 TTA"),
    return_overlay: bool = Form(default=True, description="是否返回叠加图像")
):
    """
    检测 Base64 编码图像中的裂纹
    
    - **image_base64**: Base64 编码的图像数据
    - **use_tta**: 是否使用测试时增强（TTA）
    - **return_overlay**: 是否返回叠加图像
    """
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        start_time = time.time()
        
        # 解码 Base64
        img_data = base64.b64decode(image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="无法解析图像数据")
        
        # 转换为 RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 预处理
        img_tensor = preprocess_image(img_rgb)
        
        # 推理
        with torch.no_grad():
            if use_tta and tta_inference is not None:
                pred = tta_inference(model, img_tensor.squeeze(0), device=str(device))
            else:
                outputs = model(img_tensor)
                pred = outputs['out'] if isinstance(outputs, dict) else outputs
                pred = torch.sigmoid(pred)
        
        # 后处理
        mask = pred.squeeze().cpu().numpy()
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_resized = cv2.resize(mask_binary, (img.shape[1], img.shape[0]))
        
        # 计算统计信息
        confidence = float(mask.max())
        crack_count, total_area = calculate_statistics(mask_resized)
        
        # 编码 mask
        _, mask_encoded = cv2.imencode('.png', mask_resized)
        mask_base64 = base64.b64encode(mask_encoded.tobytes()).decode('utf-8')
        
        # 叠加图像
        overlay_base64 = None
        if return_overlay:
            overlay = create_overlay(img, mask_resized)
            _, overlay_encoded = cv2.imencode('.png', overlay)
            overlay_base64 = base64.b64encode(overlay_encoded.tobytes()).decode('utf-8')
        
        processing_time = time.time() - start_time
        
        return DetectionResponse(
            success=True,
            message="检测成功",
            mask_base64=mask_base64,
            overlay_base64=overlay_base64,
            confidence=confidence,
            crack_count=crack_count,
            total_area=total_area,
            processing_time=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


def preprocess_image(img: np.ndarray, target_size: int = 512) -> torch.Tensor:
    """
    预处理图像
    
    Args:
        img: RGB 格式的 numpy 图像
        target_size: 目标尺寸
    
    Returns:
        预处理后的 tensor
    """
    # Resize
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # Normalize
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # ImageNet 标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    
    # 转换为 tensor: HWC -> CHW
    img_tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).float()
    
    # 添加 batch 维度
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor


def calculate_statistics(mask: np.ndarray) -> tuple:
    """
    计算裂纹统计信息
    
    Args:
        mask: 二值化 mask
    
    Returns:
        (crack_count, total_area)
    """
    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    
    # 计算裂纹数量（排除背景）
    crack_count = num_labels - 1
    
    # 计算总面积（像素数）
    total_area = float(np.sum(mask > 0))
    
    # 转换为面积比例
    image_area = mask.shape[0] * mask.shape[1]
    total_area_ratio = total_area / image_area
    
    return crack_count, total_area_ratio


def create_overlay(original: np.ndarray, mask: np.ndarray, 
                   color: tuple = (0, 0, 255), alpha: float = 0.5) -> np.ndarray:
    """
    创建叠加可视化图像
    
    Args:
        original: 原始图像 (BGR)
        mask: 二值化 mask
        color: 裂纹颜色 (B, G, R)
        alpha: 透明度
    
    Returns:
        叠加后的图像
    """
    overlay = original.copy()
    
    # 创建彩色 mask
    colored_mask = np.zeros_like(original)
    colored_mask[mask > 0] = color
    
    # 叠加
    mask_region = mask > 0
    overlay[mask_region] = cv2.addWeighted(
        original[mask_region], 1 - alpha,
        colored_mask[mask_region], alpha,
        0
    )
    
    return overlay


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8090))
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        workers=1
    )
