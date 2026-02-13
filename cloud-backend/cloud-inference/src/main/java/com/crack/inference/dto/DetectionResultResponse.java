package com.crack.inference.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.math.BigDecimal;
import java.time.LocalDateTime;

/**
 * 检测结果响应
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "检测结果响应")
public class DetectionResultResponse {

    @Schema(description = "结果ID")
    private Long resultId;

    @Schema(description = "任务ID")
    private Long jobId;

    @Schema(description = "任务状态")
    private String status;

    @Schema(description = "原始图像URL")
    private String imageUrl;

    @Schema(description = "掩码图像URL")
    private String maskUrl;

    @Schema(description = "叠加图像URL")
    private String overlayUrl;

    @Schema(description = "热力图URL")
    private String heatmapUrl;

    @Schema(description = "置信度")
    private BigDecimal confidence;

    @Schema(description = "裂纹数量")
    private Integer crackCount;

    @Schema(description = "裂纹总面积")
    private BigDecimal totalArea;

    @Schema(description = "处理时间（秒）")
    private BigDecimal processingTime;

    @Schema(description = "创建时间")
    private LocalDateTime createdAt;
}
