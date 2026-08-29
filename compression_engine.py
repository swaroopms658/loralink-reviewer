import torch
import numpy as np
import time
import zstandard as zstd
from typing import Tuple, Dict, Any, Optional
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedCompressionEngine:
    def __init__(self, compression_level: int = 3):
        self.compression_level = compression_level
        self.compressor = zstd.ZstdCompressor(level=compression_level)
        self.decompressor = zstd.ZstdDecompressor()
        
        self.compression_params = {
            'activations': {'sparsity_ratio': 0.3, 'quantize': False},
            'gradients': {'sparsity_ratio': 0.7, 'quantize': True},
            'labels': {'sparsity_ratio': 0.0, 'quantize': False}
        }

        import os
        if os.environ.get("LORALINK_LOSSY_COMPRESSION", "1") == "0":
            for k in self.compression_params:
                self.compression_params[k] = {"sparsity_ratio": 0.0, "quantize": False}
            logger.info("OptimizedCompressionEngine: lossy compression DISABLED "
                        "(lossless zstd only)")
        else:
            logger.info("OptimizedCompressionEngine: lossy compression ENABLED")

        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_original_bytes': 0,
            'total_compressed_bytes': 0,
            'compression_time': 0.0,
            'decompression_time': 0.0
        }
    
    def quantize_to_int8(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        assert tensor.dtype in [torch.float16, torch.float32], f"Expected float tensor, got {tensor.dtype}"
 
        tensor_flat = tensor.flatten()
        abs_values = tensor_flat.abs()

        scale_value = torch.quantile(abs_values, 0.999).item()
        if scale_value == 0:
            scale_value = abs_values.max().item()
        
        if scale_value == 0:
            return torch.zeros_like(tensor, dtype=torch.int8), 1.0, 0.0
        
        scale = scale_value / 127.0
        zero_point = 0.0
        
        quantized = torch.clamp(torch.round(tensor / scale), -128, 127).to(torch.int8)
        
        return quantized, scale, zero_point
    
    def dequantize_from_int8(self, quantized: torch.Tensor, scale: float, zero_point: float, 
                           target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        assert quantized.dtype == torch.int8, f"Expected int8 tensor, got {quantized.dtype}"
        assert target_dtype in [torch.float16, torch.float32], f"Invalid target dtype {target_dtype}"
        
        dequantized = (quantized.float() + zero_point) * scale
        return dequantized.to(target_dtype)
    
    def pack_bool_mask(self, mask: torch.Tensor) -> torch.Tensor:
        flat_mask = mask.flatten()
        mask_numpy = flat_mask.cpu().numpy().astype(np.uint8)
        
        padding_needed = (8 - len(mask_numpy) % 8) % 8
        if padding_needed > 0:
            mask_numpy = np.pad(mask_numpy, (0, padding_needed), mode='constant', constant_values=0)
        
        packed_numpy = np.packbits(mask_numpy)
        packed = torch.from_numpy(packed_numpy)
        return packed

    def unpack_bool_mask(self, packed_mask: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
        packed_numpy = packed_mask.cpu().numpy()
        unpacked_numpy = np.unpackbits(packed_numpy)
        
        original_numel = torch.tensor(original_shape).prod().item()
        unpacked_numpy = unpacked_numpy[:original_numel]
        
        unpacked = torch.from_numpy(unpacked_numpy.astype(bool))
        return unpacked.view(original_shape).to(torch.bool)
    
    def sparsify_by_magnitude(self, tensor: torch.Tensor, sparsity_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        assert 0.0 <= sparsity_ratio <= 1.0, f"Sparsity ratio must be in [0,1], got {sparsity_ratio}"
        assert tensor.numel() > 0, "Cannot sparsify empty tensor"
        
        if sparsity_ratio == 0.0:
            return tensor, torch.ones_like(tensor, dtype=torch.bool)
        
        flat_tensor = tensor.flatten()
        abs_values = flat_tensor.abs()
        
        if sparsity_ratio >= 1.0:
            return torch.zeros_like(tensor), torch.zeros_like(tensor, dtype=torch.bool)
        
        total_elements = flat_tensor.numel()
        num_to_keep = max(1, round(total_elements * (1.0 - sparsity_ratio)))
        
        
        if num_to_keep >= total_elements:
            mask = torch.ones_like(abs_values, dtype=torch.bool)
        else:
            _, topk_indices = torch.topk(abs_values, num_to_keep, largest=True)
            mask = torch.zeros_like(abs_values, dtype=torch.bool)
            mask[topk_indices] = True
        
        sparse_flat = flat_tensor * mask
        return sparse_flat.view(tensor.shape), mask.view(tensor.shape)

    def densify_sparse(self, sparse_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        assert sparse_tensor.shape == mask.shape, "Sparse tensor and mask must have same shape"
        assert mask.dtype == torch.bool, f"Mask must be boolean, got {mask.dtype}"
        return sparse_tensor
    
    def compress_tensor_data(self, data: bytes) -> bytes:
        assert len(data) > 0, "Cannot compress empty data"
        compressed = self.compressor.compress(data)
        assert len(compressed) > 0, "Compression failed"
        return compressed
    
    def decompress_tensor_data(self, compressed_data: bytes) -> bytes:
        assert len(compressed_data) > 0, "Cannot decompress empty data"
        decompressed = self.decompressor.decompress(compressed_data)
        assert len(decompressed) > 0, "Decompression failed"
        return decompressed
    
    def compress_tensor(self, tensor: torch.Tensor, tensor_type: str = 'activations') -> bytes:
        #GPU CHANGE, moving tensor to CPU first to avoid Zstd and Numpy crash on GPU tensors
        if tensor.is_cuda:
            tensor = tensor.cpu()
        start_time = time.perf_counter()
        original_dtype = tensor.dtype

        original_shape = tensor.shape
        original_size = tensor.numel() * tensor.element_size()
        
        
        if tensor_type in self.compression_params:
            params = self.compression_params[tensor_type]
        else:
            params = self.compression_params['activations']  
        
        sparsity_ratio = params['sparsity_ratio']
        should_quantize = params['quantize']
        
        logger.info(f"Starting compression ({tensor_type}): {original_shape} {original_dtype} tensor ({original_size} bytes)")
        
        if should_quantize and tensor.dtype in [torch.float16, torch.float32]:
            quantized, scale, zero_point = self.quantize_to_int8(tensor)
            is_quantized = True
        else:
            quantized, scale, zero_point = tensor, 1.0, 0.0
            is_quantized = False
        
        sparse_tensor, mask = self.sparsify_by_magnitude(quantized, sparsity_ratio)
        packed_mask = self.pack_bool_mask(mask)
        
        sparse_bytes = sparse_tensor.cpu().numpy().tobytes()
        mask_bytes = packed_mask.cpu().numpy().tobytes()

        metadata = {
            'scale': float(scale),
            'zero_point': float(zero_point),
            'original_shape': list(original_shape),
            'original_dtype': str(original_dtype),
            'is_quantized': is_quantized,
            'tensor_type': tensor_type,
            'sparsity_ratio': sparsity_ratio,
            'sparse_size': len(sparse_bytes),
            'mask_size': len(mask_bytes)
        }
        metadata_json = json.dumps(metadata).encode('utf-8')
        
        metadata_length = len(metadata_json).to_bytes(4, 'little')
        serialized_data = metadata_length + metadata_json + sparse_bytes + mask_bytes
        compressed_data = self.compress_tensor_data(serialized_data)
        
        compression_time = time.perf_counter() - start_time
        final_size = len(compressed_data)

        compression_ratio = original_size / final_size
        
        logger.info(f"Compression complete ({tensor_type}): {original_size} -> {final_size} bytes "
                   f"({compression_ratio:.2f}x, sparsity={sparsity_ratio:.1%}, quantized={is_quantized}) "
                   f"in {compression_time:.6f}s")
        
        self.stats['total_compressions'] += 1
        self.stats['total_original_bytes'] += original_size
        self.stats['total_compressed_bytes'] += final_size
        self.stats['compression_time'] += compression_time
        
        return compressed_data
    
    def decompress_tensor(self, compressed_data: bytes) -> torch.Tensor:
        assert isinstance(compressed_data, bytes), f"Expected bytes, got {type(compressed_data)}"
        start_time = time.perf_counter()
        
        logger.info(f"Starting decompression: {len(compressed_data)} bytes")

        
        serialized_data = self.decompress_tensor_data(compressed_data)
        
        if len(serialized_data) < 4:
            raise ValueError(f"Invalid compressed data: too short ({len(serialized_data)} bytes)")
        
        metadata_length = int.from_bytes(serialized_data[:4], 'little')
        if metadata_length <= 0 or metadata_length > len(serialized_data) - 4:
            raise ValueError(f"Invalid metadata length: {metadata_length}")
        
        metadata_end = 4 + metadata_length
        
        try:
            metadata_json = serialized_data[4:metadata_end].decode('utf-8')
            metadata = json.loads(metadata_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse metadata: {e}")

        required_keys = ['sparse_size', 'mask_size', 'original_shape', 'original_dtype', 'scale', 'zero_point', 'is_quantized']
        for key in required_keys:
            if key not in metadata:
                raise ValueError(f"Missing required metadata key: {key}")

        sparse_size = metadata['sparse_size']
        mask_size = metadata['mask_size']
        sparse_end = metadata_end + sparse_size
        
        if sparse_end + mask_size != len(serialized_data):
            raise ValueError(f"Data size mismatch in decompression")

        sparse_bytes = serialized_data[metadata_end:sparse_end]
        mask_bytes = serialized_data[sparse_end:sparse_end + mask_size]

        original_shape = tuple(metadata['original_shape'])
        original_dtype_str = metadata['original_dtype']
        scale = metadata['scale']
        zero_point = metadata['zero_point']
        is_quantized = metadata['is_quantized']
        tensor_type = metadata.get('tensor_type', 'unknown')
        
        dtype_map = {
            'torch.int64': np.int64,
            'torch.int32': np.int32,
            'torch.int16': np.int16,
            'torch.int8': np.int8,
            'torch.uint8': np.uint8,
            'torch.float64': np.float64,
            'torch.float32': np.float32,
            'torch.float16': np.float16,
            'torch.bool': np.bool_
        }
        
        if is_quantized:
            numpy_dtype = np.int8
        elif original_dtype_str in dtype_map:
            numpy_dtype = dtype_map[original_dtype_str]
        else:
            raise ValueError(f"Unsupported dtype: {original_dtype_str}")
        
        try:
            sparse_array = np.frombuffer(sparse_bytes, dtype=numpy_dtype).copy()
            mask_array = np.frombuffer(mask_bytes, dtype=np.uint8).copy()
        except ValueError as e:
            raise ValueError(f"Failed to reconstruct arrays: {e}")
        
        expected_elements = np.prod(original_shape)
        if len(sparse_array) != expected_elements:
            raise ValueError(f"Sparse array size mismatch: expected {expected_elements}, got {len(sparse_array)}")
        
        sparse_tensor = torch.from_numpy(sparse_array).view(original_shape)
        packed_mask = torch.from_numpy(mask_array)
        mask = self.unpack_bool_mask(packed_mask, original_shape)
        
        dense_tensor = self.densify_sparse(sparse_tensor, mask)
        
        if is_quantized:
            target_dtype = torch.float16 if original_dtype_str == 'torch.float16' else torch.float32
            final_tensor = self.dequantize_from_int8(dense_tensor, scale, zero_point, target_dtype)
        else:
            final_tensor = dense_tensor
        
        decompression_time = time.perf_counter() - start_time
        
        logger.info(f"Decompression complete ({tensor_type}): {final_tensor.shape} {final_tensor.dtype} "

                   f"in {decompression_time:.6f}s")
        
        self.stats['total_decompressions'] += 1
        self.stats['decompression_time'] += decompression_time
        
        if final_tensor.shape != tuple(original_shape):
            raise ValueError(f"Shape mismatch after decompression")
        
        return final_tensor
    
    def get_compression_stats(self) -> Dict[str, Any]:
        if self.stats['total_compressions'] == 0:
            return {'status': 'No compressions performed yet'}
        
        avg_compression_ratio = (self.stats['total_original_bytes'] / 
                               self.stats['total_compressed_bytes'])
        
        return {
            'total_compressions': self.stats['total_compressions'],
            'total_decompressions': self.stats['total_decompressions'],
            'average_compression_ratio': f"{avg_compression_ratio:.2f}x",
            'total_bytes_saved': (self.stats['total_original_bytes'] - 
                                self.stats['total_compressed_bytes']),
            'avg_compression_time': (self.stats['compression_time'] / 
                                   self.stats['total_compressions']),
            'avg_decompression_time': (self.stats['decompression_time'] / 
                                     max(1, self.stats['total_decompressions'])),
            'compression_efficiency': f"{(1 - self.stats['total_compressed_bytes'] / self.stats['total_original_bytes']) * 100:.1f}%"
        }