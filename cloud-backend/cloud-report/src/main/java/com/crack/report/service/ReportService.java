package com.crack.report.service;

import com.crack.common.result.PageResult;
import com.crack.report.dto.GenerateReportRequest;
import com.crack.report.dto.ReportResponse;

/**
 * 报告服务接口
 */
public interface ReportService {

    /**
     * 生成报告
     *
     * @param request 生成请求
     * @param userId  用户ID
     * @return 报告响应
     */
    ReportResponse generateReport(GenerateReportRequest request, Long userId);

    /**
     * 获取报告详情
     *
     * @param reportId 报告ID
     * @param userId   用户ID
     * @return 报告响应
     */
    ReportResponse getReport(Long reportId, Long userId);

    /**
     * 获取报告列表
     *
     * @param page   页码
     * @param size   每页条数
     * @param userId 用户ID
     * @return 分页结果
     */
    PageResult<ReportResponse> listReports(Integer page, Integer size, Long userId);

    /**
     * 删除报告
     *
     * @param reportId 报告ID
     * @param userId   用户ID
     */
    void deleteReport(Long reportId, Long userId);

    /**
     * 获取报告下载URL
     *
     * @param reportId 报告ID
     * @param userId   用户ID
     * @return 下载URL
     */
    String getDownloadUrl(Long reportId, Long userId);
}
