package com.crack.dataset.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import lombok.Data;

/**
 * 创建数据集请求
 */
@Data
@Schema(description = "创建数据集请求")
public class CreateDatasetRequest {

    @NotBlank(message = "数据集名称不能为空")
    @Schema(description = "数据集名称", example = "CrackTree260")
    private String name;

    @Schema(description = "数据集描述")
    private String description;

    @Schema(description = "数据集格式", example = "CUSTOM")
    private String format = "CUSTOM";
}
