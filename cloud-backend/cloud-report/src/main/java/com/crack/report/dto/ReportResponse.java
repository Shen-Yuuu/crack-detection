package com.crack.report.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * 报告响应
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "报告响应")
public class ReportResponse {

    @Schema(description = "报告ID")
    private Long id;

    @Schema(description = "报告标题")
    private String title;

    @Schema(description = "报告类型")
    private String reportType;

    @Schema(description = "报告文件URL")
    private String fileUrl;

    @Schema(description = "状态")
    private String status;

    @Schema(description = "创建时间")
    private LocalDateTime createdAt;
}
