package com.crack.dataset.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 数据集查询参数
 */
@Data
@Schema(description = "数据集查询参数")
public class DatasetQueryParam {

    @Schema(description = "页码", example = "1")
    private Integer page = 1;

    @Schema(description = "每页条数", example = "10")
    private Integer size = 10;

    @Schema(description = "数据集名称（模糊搜索）")
    private String name;

    @Schema(description = "状态")
    private String status;
}
