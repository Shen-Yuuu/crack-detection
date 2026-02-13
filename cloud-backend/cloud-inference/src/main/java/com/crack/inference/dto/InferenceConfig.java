package com.crack.inference.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

import java.math.BigDecimal;

/**
 * 推理配置
 */
@Data
@Schema(description = "推理配置")
public class InferenceConfig {

    @Schema(description = "检测阈值", example = "0.5")
    private BigDecimal threshold = new BigDecimal("0.5");

    @Schema(description = "是否使用TTA", example = "false")
    private Boolean useTta = false;

    @Schema(description = "最小面积过滤", example = "100")
    private Integer minArea = 100;

    @Schema(description = "是否返回掩码图", example = "true")
    private Boolean returnMask = true;

    @Schema(description = "是否返回叠加图", example = "true")
    private Boolean returnOverlay = true;
}
