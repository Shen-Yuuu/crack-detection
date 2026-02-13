package com.crack.inference.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;

/**
 * Python推理服务响应
 */
@Data
@Schema(description = "Python推理服务响应")
public class PythonInferenceResponse {

    @Schema(description = "是否成功")
    private Boolean success;

    @Schema(description = "消息")
    private String message;

    @Schema(description = "掩码图Base64")
    @JsonProperty("mask_base64")
    private String maskBase64;

    @Schema(description = "叠加图Base64")
    @JsonProperty("overlay_base64")
    private String overlayBase64;

    @Schema(description = "置信度")
    private BigDecimal confidence;

    @Schema(description = "处理时间（秒）")
    @JsonProperty("processing_time")
    private BigDecimal processingTime;

    @Schema(description = "裂纹数量")
    @JsonProperty("crack_count")
    private Integer crackCount;

    @Schema(description = "裂纹总面积")
    @JsonProperty("total_area")
    private BigDecimal totalArea;
}
