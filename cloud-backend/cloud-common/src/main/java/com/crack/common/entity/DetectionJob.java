package com.crack.common.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 检测任务实体
 */
@Data
@EqualsAndHashCode(callSuper = true)
@TableName("detection_jobs")
public class DetectionJob extends BaseEntity {

    /**
     * 用户ID
     */
    private Long userId;

    /**
     * 原始图像URL
     */
    private String imageUrl;

    /**
     * 图像名称
     */
    private String imageName;

    /**
     * 模型版本
     */
    private String modelVersion;

    /**
     * 推理配置（JSON）
     */
    private String config;

    /**
     * 状态：pending, running, completed, failed
     */
    private String status;

    /**
     * 结果ID
     */
    private Long resultId;

    /**
     * 错误消息
     */
    private String errorMessage;
}
