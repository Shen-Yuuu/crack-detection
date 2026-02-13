package com.crack.common.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 数据集实体
 */
@Data
@EqualsAndHashCode(callSuper = true)
@TableName("datasets")
public class Dataset extends BaseEntity {

    /**
     * 数据集名称
     */
    private String name;

    /**
     * 描述
     */
    private String description;

    /**
     * 数据集格式：COCO, VOC, YOLO, CUSTOM
     */
    private String format;

    /**
     * 存储路径
     */
    private String storagePath;

    /**
     * 总图像数
     */
    private Integer totalImages;

    /**
     * 训练集数量
     */
    private Integer trainCount;

    /**
     * 验证集数量
     */
    private Integer valCount;

    /**
     * 测试集数量
     */
    private Integer testCount;

    /**
     * 状态：pending, processing, completed, failed
     */
    private String status;

    /**
     * 所属用户ID
     */
    private Long userId;
}
