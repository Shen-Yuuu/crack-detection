package com.crack.common.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 图像实体
 */
@Data
@EqualsAndHashCode(callSuper = true)
@TableName("images")
public class Image extends BaseEntity {

    /**
     * 所属数据集ID
     */
    private Long datasetId;

    /**
     * 文件名
     */
    private String fileName;

    /**
     * 文件存储路径
     */
    private String filePath;

    /**
     * 掩码存储路径
     */
    private String maskPath;

    /**
     * 图像宽度
     */
    private Integer width;

    /**
     * 图像高度
     */
    private Integer height;

    /**
     * 文件大小（字节）
     */
    private Long fileSize;

    /**
     * 数据集划分：train, val, test
     */
    private String split;
}
