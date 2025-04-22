#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GPU资源占用程序 - 加载Qwen2.5-72B-Instruct模型
纯粹用于占用指定GPU的显存
"""

# 在导入其他库之前首先处理GPU设备选择
import os
import sys
import argparse

# 先解析参数获取GPU设备ID
parser = argparse.ArgumentParser(description="GPU资源占用工具 - 加载大型语言模型")
parser.add_argument(
    "--gpus", 
    type=str,
    default=None, 
    help="指定要使用的GPU设备ID，例如 '5,6'"
)
# 只解析gpus参数，其他参数留到后面处理
args, _ = parser.parse_known_args()

# 如果指定了GPU，立即设置环境变量
if args.gpus:
    device_ids = [id.strip() for id in args.gpus.split(",")]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(device_ids)
    print(f"已设置CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"程序将只使用指定的GPU: {args.gpus}")

# 导入其他必要的库
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import importlib.util

def is_package_available(package_name):
    """检查是否安装了指定的包"""
    return importlib.util.find_spec(package_name) is not None

def get_gpu_memory_info():
    """获取GPU内存使用情况"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        memory_info = []
        
        for i in range(gpu_count):
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
            reserved_memory = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
            free_memory = total_memory - allocated_memory
            
            memory_info.append({
                "设备": i, 
                "总内存(GB)": round(total_memory, 2),
                "已分配(GB)": round(allocated_memory, 2),
                "已预留(GB)": round(reserved_memory, 2),
                "可用(GB)": round(free_memory, 2)
            })
        
        return memory_info
    else:
        return "CUDA不可用"

def print_memory_info(memory_info):
    """打印内存使用情况"""
    if isinstance(memory_info, list):
        print("\n===== GPU内存使用情况 =====")
        for device in memory_info:
            print(f"GPU {device['设备']}:")
            print(f"  总内存: {device['总内存(GB)']:.2f} GB")
            print(f"  已分配: {device['已分配(GB)']:.2f} GB ({device['已分配(GB)']/device['总内存(GB)']*100:.1f}%)")
            print(f"  已预留: {device['已预留(GB)']:.2f} GB ({device['已预留(GB)']/device['总内存(GB)']*100:.1f}%)")
            print(f"  可用内存: {device['可用(GB)']:.2f} GB ({device['可用(GB)']/device['总内存(GB)']*100:.1f}%)")
            print("-" * 30)
    else:
        print(memory_info)

def load_model(model_name="Qwen/Qwen2.5-72B-Instruct", use_flash_attn=True, dtype=torch.bfloat16):
    """加载大型LLM模型到指定GPU上，不使用量化，最大化显存占用"""
    print(f"\n正在加载模型: {model_name}")
    print(f"使用类型: {dtype}")
    print(f"Flash Attention: {use_flash_attn}")

    # 检查是否安装了flash_attn
    flash_attn_available = is_package_available("flash_attn")
    if use_flash_attn and not flash_attn_available:
        print("\n警告: 您启用了Flash Attention 2，但未安装flash_attn包。")
        print("将禁用Flash Attention功能继续加载模型。")
        print("如需使用Flash Attention 2，请参考: https://huggingface.co/docs/transformers/perf_infer_gpu_one#flashattention-2")
        use_flash_attn = False

    # 加载前先清理GPU缓存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 打印加载前的内存使用情况
    print("\n加载前GPU内存状态:")
    print_memory_info(get_gpu_memory_info())

    start_time = time.time()

    try:
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )

        # 准备模型加载参数 - 不使用量化，最大化显存占用
        model_kwargs = {
            "device_map": "auto",  # 自动在可见GPU上分配
            "trust_remote_code": True,
            "torch_dtype": dtype,  # 使用指定精度
        }
        
        # 使用新的attn_implementation参数而不是弃用的use_flash_attention_2
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        load_time = time.time() - start_time
        print(f"\n模型加载完成，用时：{load_time:.2f}秒")
        
        # 打印加载后的内存使用情况
        print("\n加载后GPU内存状态:")
        print_memory_info(get_gpu_memory_info())
        
        # 打印模型分布情况
        if hasattr(model, "hf_device_map"):
            print("\n模型层分布情况:")
            for layer, device in model.hf_device_map.items():
                print(f"  {layer}: {device}")
        
        return tokenizer, model
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        print("\n如果错误与Flash Attention相关，可以尝试使用--no-flash-attn选项禁用Flash Attention。")
        return None, None

def run_interactive_session(tokenizer, model):
    """运行一个简单的交互会话，保持程序运行"""
    print("\n模型已加载并占用GPU资源。按Ctrl+C退出。")
    print("你可以输入文本进行推理测试，输入'memory'查看当前内存使用情况，或输入'exit'退出。")
    
    try:
        while True:
            user_input = input("\n>>> ")
            
            if user_input.lower() == 'exit':
                break
                
            if user_input.lower() == 'memory':
                print_memory_info(get_gpu_memory_info())
                continue
            
            # 简单的推理测试
            inputs = tokenizer(user_input, return_tensors="pt")
            # 将输入移动到模型所在设备
            if hasattr(model, "device"):
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # 记录开始时间
            start_time = time.time()
            
            # 生成回复
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
                
            # 计算耗时
            inference_time = time.time() - start_time
            
            # 解码输出
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n[推理耗时: {inference_time:.2f}秒]")
            print(f"回复: {response}")
    except KeyboardInterrupt:
        print("\n退出程序...")
    finally:
        # 清理资源
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("已释放GPU资源")

def parse_arguments():
    """解析命令行参数"""
    # 重新创建一个解析器，包含所有参数
    parser = argparse.ArgumentParser(description="GPU资源占用工具 - 加载大型语言模型")
    parser.add_argument(
        "--model", 
        type=str, 
        default="Qwen/Qwen2.5-72B-Instruct", 
        help="要加载的模型名称或路径"
    )
    parser.add_argument(
        "--gpus", 
        type=str,
        default=None, 
        help="指定要使用的GPU设备ID，例如 '5,6'"
    )
    parser.add_argument(
        "--fp16", 
        action="store_true", 
        help="使用float16精度加载模型"
    )
    parser.add_argument(
        "--fp32", 
        action="store_true", 
        help="使用float32精度加载模型 (占用最多显存)"
    )
    parser.add_argument(
        "--no-flash-attn", 
        action="store_true", 
        help="禁用Flash Attention"
    )
    parser.add_argument(
        "--install-flash-attn",
        action="store_true",
        help="尝试安装flash_attn包"
    )
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_arguments()
    
    # 如果用户要求安装flash_attn
    if args.install_flash_attn:
        print("正在尝试安装flash_attn包...")
        import subprocess
        try:
            subprocess.check_call([
                "pip", "install", "flash-attn", "--no-build-isolation"
            ])
            print("flash_attn安装完成。")
        except subprocess.CalledProcessError:
            print("flash_attn安装失败，请手动安装。参考: https://github.com/Dao-AILab/flash-attention")
    
    # 确定使用的精度
    if args.fp32:
        dtype = torch.float32
    elif args.fp16:
        dtype = torch.float16
    else:
        dtype = torch.bfloat16  # 默认使用bfloat16，平衡显存占用和精度
    
    use_flash_attn = not args.no_flash_attn
    
    # 显示CUDA可用性信息
    if torch.cuda.is_available():
        print(f"CUDA可用，设备总数: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  设备 {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA不可用，请确认是否安装GPU版本的PyTorch。")
        return
    
    # 加载模型
    tokenizer, model = load_model(
        model_name=args.model,
        use_flash_attn=use_flash_attn,
        dtype=dtype
    )
    
    if model is not None:
        run_interactive_session(tokenizer, model)

if __name__ == "__main__":
    main()
