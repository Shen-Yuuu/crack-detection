package com.crack.visual.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 叠加配置
 */
@Data
@Schema(description = "叠加配置")
public class OverlayConfig {

    @Schema(description = "透明度", example = "0.5")
    private Float alpha = 0.5f;

    @Schema(description = "掩码颜色（十六进制）", example = "#FF0000")
    private String maskColor = "#FF0000";

    @Schema(description = "是否绘制轮廓", example = "true")
    private Boolean drawContours = true;

    @Schema(description = "轮廓颜色", example = "#00FF00")
    private String contourColor = "#00FF00";

    @Schema(description = "轮廓线宽", example = "2")
    private Integer contourWidth = 2;
}
