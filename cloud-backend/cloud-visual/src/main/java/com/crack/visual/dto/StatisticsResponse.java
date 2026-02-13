package com.crack.visual.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.util.Map;

/**
 * 统计响应
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "统计响应")
public class StatisticsResponse {

    @Schema(description = "裂纹总数")
    private Integer totalCracks;

    @Schema(description = "总长度（米）")
    private BigDecimal totalLength;

    @Schema(description = "平均宽度（毫米）")
    private BigDecimal avgWidth;

    @Schema(description = "最大宽度（毫米）")
    private BigDecimal maxWidth;

    @Schema(description = "总面积（平方米）")
    private BigDecimal totalArea;

    @Schema(description = "裂纹类型分布")
    private Map<String, Integer> typeDistribution;

    @Schema(description = "严重程度分布")
    private Map<String, Integer> severityDistribution;
}
