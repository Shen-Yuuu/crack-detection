package com.crack.report.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import lombok.Data;

import java.util.List;

/**
 * 生成报告请求
 */
@Data
@Schema(description = "生成报告请求")
public class GenerateReportRequest {

    @NotBlank(message = "报告标题不能为空")
    @Schema(description = "报告标题", example = "道路裂纹检测报告")
    private String title;

    @NotEmpty(message = "结果ID列表不能为空")
    @Schema(description = "检测结果ID列表")
    private List<Long> resultIds;

    @Schema(description = "报告类型：pdf, excel", example = "pdf")
    private String reportType = "pdf";

    @Schema(description = "是否包含图像", example = "true")
    private Boolean includeImages = true;

    @Schema(description = "是否包含统计图表", example = "true")
    private Boolean includeCharts = true;
}
