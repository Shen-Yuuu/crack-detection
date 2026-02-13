package com.crack.inference.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.util.List;

/**
 * 批量推理请求
 */
@Data
@Schema(description = "批量推理请求")
public class BatchInferenceRequest {

    @Schema(description = "图像URL列表")
    private List<String> imageUrls;

    @Schema(description = "模型版本", example = "v1.0")
    private String modelVersion = "v1.0";

    @Schema(description = "推理配置")
    private InferenceConfig config;
}
