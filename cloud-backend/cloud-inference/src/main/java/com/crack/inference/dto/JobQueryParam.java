package com.crack.inference.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.Data;

/**
 * 任务查询参数
 */
@Data
@Schema(description = "任务查询参数")
public class JobQueryParam {

    @Schema(description = "页码", example = "1")
    private Integer page = 1;

    @Schema(description = "每页条数", example = "10")
    private Integer size = 10;

    @Schema(description = "任务状态")
    private String status;

    @Schema(description = "模型版本")
    private String modelVersion;
}
