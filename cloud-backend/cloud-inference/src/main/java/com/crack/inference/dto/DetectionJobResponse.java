package com.crack.inference.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * 检测任务响应
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "检测任务响应")
public class DetectionJobResponse {

    @Schema(description = "任务ID")
    private Long jobId;

    @Schema(description = "任务状态")
    private String status;

    @Schema(description = "原始图像URL")
    private String imageUrl;

    @Schema(description = "模型版本")
    private String modelVersion;

    @Schema(description = "创建时间")
    private LocalDateTime createdAt;

    @Schema(description = "错误消息")
    private String errorMessage;
}
