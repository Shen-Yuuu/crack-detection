"""
模型导出到ONNX和TensorRT优化
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).parent))

from models.convnext_upernet import create_model


def export_to_onnx(model: nn.Module,
                   output_path: str,
                   input_shape: tuple = (1, 3, 512, 512),
                   opset_version: int = 17,
                   dynamic_axes: bool = True):
    """
    导出模型到ONNX格式
    
    Args:
        model: PyTorch模型
        output_path: 输出路径
        input_shape: 输入形状 (B, C, H, W)
        opset_version: ONNX opset版本
        dynamic_axes: 是否使用动态轴
    """
    model.eval()
    
    # 创建示例输入
    dummy_input = torch.randn(*input_shape)
    
    # 动态轴配置
    if dynamic_axes:
        dynamic_axes_config = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    else:
        dynamic_axes_config = None
    
    # 导出
    print(f"Exporting model to ONNX...")
    print(f"Input shape: {input_shape}")
    print(f"Opset version: {opset_version}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_config,
        verbose=False
    )
    
    print(f"Model exported to: {output_path}")
    
    # 验证ONNX模型
    print("\nValidating ONNX model...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")
    
    return output_path


def verify_onnx(pytorch_model: nn.Module,
                onnx_path: str,
                input_shape: tuple = (1, 3, 512, 512),
                tolerance: float = 1e-5):
    """
    验证ONNX模型输出与PyTorch模型一致
    
    Args:
        pytorch_model: PyTorch模型
        onnx_path: ONNX模型路径
        input_shape: 输入形状
        tolerance: 误差容忍度
    """
    print("\nVerifying ONNX model accuracy...")
    
    pytorch_model.eval()
    
    # 创建测试输入
    test_input = torch.randn(*input_shape)
    
    # PyTorch推理
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
        if isinstance(pytorch_output, dict):
            pytorch_output = pytorch_output['out']
        pytorch_output = torch.sigmoid(pytorch_output).numpy()
    
    # ONNX推理
    ort_session = ort.InferenceSession(onnx_path)
    onnx_input = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, onnx_input)[0]
    onnx_output = 1 / (1 + np.exp(-onnx_output))  # Sigmoid
    
    # 比较输出
    diff = np.abs(pytorch_output - onnx_output)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.8f}")
    print(f"Mean difference: {mean_diff:.8f}")
    
    if max_diff < tolerance:
        print(f"✓ Verification passed! (tolerance: {tolerance})")
    else:
        print(f"✗ Verification failed! Max diff {max_diff} > tolerance {tolerance}")
    
    return max_diff < tolerance


def quantize_onnx(onnx_path: str,
                  output_path: str,
                  quantization_mode: str = 'dynamic'):
    """
    量化ONNX模型
    
    Args:
        onnx_path: 输入ONNX模型路径
        output_path: 输出量化模型路径
        quantization_mode: 量化模式 ('dynamic', 'static')
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
        from onnxruntime.quantization import CalibrationDataReader
    except ImportError:
        print("onnxruntime quantization not available")
        return None
    
    print(f"\nQuantizing ONNX model ({quantization_mode})...")
    
    if quantization_mode == 'dynamic':
        # 动态量化（无需校准数据）
        quantize_dynamic(
            onnx_path,
            output_path,
            weight_type=QuantType.QUInt8
        )
    else:
        print("Static quantization requires calibration data")
        return None
    
    print(f"Quantized model saved to: {output_path}")
    
    # 比较模型大小
    original_size = Path(onnx_path).stat().st_size / (1024 * 1024)
    quantized_size = Path(output_path).stat().st_size / (1024 * 1024)
    
    print(f"Original size: {original_size:.2f} MB")
    print(f"Quantized size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
    
    return output_path


def optimize_onnx(onnx_path: str, output_path: str):
    """
    优化ONNX模型（常数折叠、算子融合等）
    
    Args:
        onnx_path: 输入ONNX模型路径
        output_path: 输出优化模型路径
    """
    try:
        from onnxruntime.transformers.optimizer import optimize_model
    except ImportError:
        print("onnxruntime optimizer not available, using basic optimization")
        
        import onnx
        from onnx import optimizer
        
        model = onnx.load(onnx_path)
        
        # 基础优化
        passes = [
            'eliminate_identity',
            'eliminate_nop_dropout',
            'eliminate_nop_pad',
            'eliminate_nop_transpose',
            'eliminate_unused_initializer',
            'fuse_bn_into_conv',
            'fuse_consecutive_transposes',
        ]
        
        optimized_model = optimizer.optimize(model, passes)
        onnx.save(optimized_model, output_path)
        
        print(f"Optimized model saved to: {output_path}")
        return output_path
    
    print("\nOptimizing ONNX model...")
    
    optimized_model = optimize_model(
        onnx_path,
        model_type='bert',  # 通用优化
        num_heads=0,
        hidden_size=0
    )
    
    optimized_model.save_model_to_file(output_path)
    print(f"Optimized model saved to: {output_path}")
    
    return output_path


def benchmark_onnx(onnx_path: str,
                   input_shape: tuple = (1, 3, 512, 512),
                   num_iterations: int = 100):
    """
    性能基准测试
    
    Args:
        onnx_path: ONNX模型路径
        input_shape: 输入形状
        num_iterations: 测试迭代次数
    """
    print(f"\nBenchmarking ONNX model...")
    print(f"Input shape: {input_shape}")
    print(f"Iterations: {num_iterations}")
    
    # 创建会话
    ort_session = ort.InferenceSession(onnx_path)
    
    # 预热
    dummy_input = {ort_session.get_inputs()[0].name: np.random.randn(*input_shape).astype(np.float32)}
    for _ in range(10):
        ort_session.run(None, dummy_input)
    
    # 基准测试
    import time
    
    times = []
    for _ in range(num_iterations):
        start = time.time()
        ort_session.run(None, dummy_input)
        times.append(time.time() - start)
    
    times = np.array(times) * 1000  # 转换为毫秒
    
    print(f"Mean: {np.mean(times):.2f} ms")
    print(f"Std: {np.std(times):.2f} ms")
    print(f"Min: {np.min(times):.2f} ms")
    print(f"Max: {np.max(times):.2f} ms")
    print(f"Median: {np.median(times):.2f} ms")


def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='model.onnx',
                       help='Output ONNX path')
    parser.add_argument('--input-shape', type=int, nargs=4, default=[1, 3, 512, 512],
                       help='Input shape (B C H W)')
    parser.add_argument('--opset', type=int, default=17,
                       help='ONNX opset version')
    parser.add_argument('--verify', action='store_true',
                       help='Verify ONNX model')
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize ONNX model')
    parser.add_argument('--optimize', action='store_true',
                       help='Optimize ONNX model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark ONNX model')
    
    args = parser.parse_args()
    
    # 加载PyTorch模型
    print("Loading PyTorch model...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    # 推断模型配置（简化版，实际应从config读取）
    model_config = {
        'backbone': 'convnext_tiny',
        'pretrained': False,
        'num_classes': 1,
        'decoder_channels': 256,
        'deep_supervision': False,  # 导出时关闭深度监督
        'edge_branch': False  # 导出时关闭边界分支
    }
    
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print("Model loaded successfully!")
    
    # 导出ONNX
    input_shape = tuple(args.input_shape)
    onnx_path = export_to_onnx(
        model,
        args.output,
        input_shape=input_shape,
        opset_version=args.opset
    )
    
    # 验证
    if args.verify:
        verify_onnx(model, onnx_path, input_shape)
    
    # 优化
    if args.optimize:
        optimized_path = args.output.replace('.onnx', '_optimized.onnx')
        optimize_onnx(onnx_path, optimized_path)
        onnx_path = optimized_path
    
    # 量化
    if args.quantize:
        quantized_path = args.output.replace('.onnx', '_quantized.onnx')
        quantize_onnx(onnx_path, quantized_path)
    
    # 基准测试
    if args.benchmark:
        benchmark_onnx(onnx_path, input_shape)
    
    print("\n" + "=" * 60)
    print("Export completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

