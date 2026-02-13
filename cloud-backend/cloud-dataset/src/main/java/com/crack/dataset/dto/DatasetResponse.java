package com.crack.dataset.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * 数据集响应
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "数据集响应")
public class DatasetResponse {

    @Schema(description = "数据集ID")
    private Long id;

    @Schema(description = "数据集名称")
    private String name;

    @Schema(description = "描述")
    private String description;

    @Schema(description = "格式")
    private String format;

    @Schema(description = "总图像数")
    private Integer totalImages;

    @Schema(description = "训练集数量")
    private Integer trainCount;

    @Schema(description = "验证集数量")
    private Integer valCount;

    @Schema(description = "测试集数量")
    private Integer testCount;

    @Schema(description = "状态")
    private String status;

    @Schema(description = "创建时间")
    private LocalDateTime createdAt;
}
