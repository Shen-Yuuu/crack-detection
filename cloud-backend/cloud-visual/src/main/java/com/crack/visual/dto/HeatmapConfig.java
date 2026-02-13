package com.crack.visual.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 热力图配置
 */
@Data
@Schema(description = "热力图配置")
public class HeatmapConfig {

    @Schema(description = "颜色映射", example = "jet")
    private String colorMap = "jet";

    @Schema(description = "透明度", example = "0.7")
    private Float alpha = 0.7f;

    @Schema(description = "是否显示颜色条", example = "true")
    private Boolean showColorBar = true;
}
