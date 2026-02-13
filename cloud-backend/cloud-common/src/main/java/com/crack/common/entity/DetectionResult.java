package com.crack.common.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;

import java.math.BigDecimal;

/**
 * 检测结果实体
 */
@Data
@EqualsAndHashCode(callSuper = true)
@TableName("detection_results")
public class DetectionResult extends BaseEntity {

    /**
     * 关联任务ID
     */
    private Long jobId;

    /**
     * 掩码图像URL
     */
    private String maskUrl;

    /**
     * 叠加图像URL
     */
    private String overlayUrl;

    /**
     * 热力图URL
     */
    private String heatmapUrl;

    /**
     * 裂纹向量数据（JSON）
     */
    private String vectors;

    /**
     * 裂纹属性数据（JSON）
     */
    private String attributes;

    /**
     * 置信度
     */
    private BigDecimal confidence;

    /**
     * 处理时间（秒）
     */
    private BigDecimal processingTime;

    /**
     * 裂纹数量
     */
    private Integer crackCount;

    /**
     * 裂纹总面积
     */
    private BigDecimal totalArea;
}
