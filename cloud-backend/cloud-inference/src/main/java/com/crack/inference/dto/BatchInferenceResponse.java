package com.crack.inference.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * 批量推理响应
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "批量推理响应")
public class BatchInferenceResponse {

    @Schema(description = "批次ID")
    private String batchId;

    @Schema(description = "任务总数")
    private Integer totalJobs;

    @Schema(description = "任务ID列表")
    private List<Long> jobIds;

    @Schema(description = "状态")
    private String status;
}
