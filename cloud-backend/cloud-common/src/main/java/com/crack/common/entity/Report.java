package com.crack.common.entity;

import com.baomidou.mybatisplus.annotation.TableName;
import lombok.Data;
import lombok.EqualsAndHashCode;

/**
 * 报告实体
 */
@Data
@EqualsAndHashCode(callSuper = true)
@TableName("reports")
public class Report extends BaseEntity {

    /**
     * 用户ID
     */
    private Long userId;

    /**
     * 报告标题
     */
    private String title;

    /**
     * 报告类型：pdf, excel
     */
    private String reportType;

    /**
     * 报告文件URL
     */
    private String fileUrl;

    /**
     * 关联的结果ID列表（JSON数组）
     */
    private String resultIds;

    /**
     * 状态：generating, completed, failed
     */
    private String status;
}
