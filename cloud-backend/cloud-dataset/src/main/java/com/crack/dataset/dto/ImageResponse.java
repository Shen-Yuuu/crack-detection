package com.crack.dataset.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * 图像响应
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@Schema(description = "图像响应")
public class ImageResponse {

    @Schema(description = "图像ID")
    private Long id;

    @Schema(description = "数据集ID")
    private Long datasetId;

    @Schema(description = "文件名")
    private String fileName;

    @Schema(description = "图像URL")
    private String imageUrl;

    @Schema(description = "掩码URL")
    private String maskUrl;

    @Schema(description = "宽度")
    private Integer width;

    @Schema(description = "高度")
    private Integer height;

    @Schema(description = "文件大小（字节）")
    private Long fileSize;

    @Schema(description = "数据集划分")
    private String split;

    @Schema(description = "创建时间")
    private LocalDateTime createdAt;
}
