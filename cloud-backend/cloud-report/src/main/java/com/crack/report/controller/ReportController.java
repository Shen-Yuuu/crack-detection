package com.crack.report.controller;

import com.crack.common.result.PageResult;
import com.crack.common.result.Result;
import com.crack.report.dto.GenerateReportRequest;
import com.crack.report.dto.ReportResponse;
import com.crack.report.service.ReportService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

/**
 * 报告控制器
 */
@RestController
@RequestMapping("/api/v1/report")
@RequiredArgsConstructor
@Tag(name = "报告管理", description = "报告生成与下载接口")
public class ReportController {

    private final ReportService reportService;

    @PostMapping("/generate")
    @Operation(summary = "生成报告", description = "根据检测结果生成报告")
    public Result<ReportResponse> generateReport(
            @Valid @RequestBody GenerateReportRequest request,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        ReportResponse response = reportService.generateReport(request, userId);
        return Result.success("报告生成任务已提交", response);
    }

    @GetMapping("/{reportId}")
    @Operation(summary = "获取报告", description = "获取报告详情")
    public Result<ReportResponse> getReport(
            @Parameter(description = "报告ID") @PathVariable Long reportId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        ReportResponse response = reportService.getReport(reportId, userId);
        return Result.success(response);
    }

    @GetMapping
    @Operation(summary = "报告列表", description = "获取报告列表")
    public Result<PageResult<ReportResponse>> listReports(
            @Parameter(description = "页码") @RequestParam(value = "page", defaultValue = "1") Integer page,
            @Parameter(description = "每页条数") @RequestParam(value = "size", defaultValue = "10") Integer size,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        PageResult<ReportResponse> result = reportService.listReports(page, size, userId);
        return Result.success(result);
    }

    @DeleteMapping("/{reportId}")
    @Operation(summary = "删除报告", description = "删除报告")
    public Result<Void> deleteReport(
            @Parameter(description = "报告ID") @PathVariable Long reportId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        reportService.deleteReport(reportId, userId);
        return Result.success("删除成功", null);
    }

    @GetMapping("/{reportId}/download")
    @Operation(summary = "下载报告", description = "获取报告下载链接")
    public Result<String> downloadReport(
            @Parameter(description = "报告ID") @PathVariable Long reportId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        String url = reportService.getDownloadUrl(reportId, userId);
        return Result.success(url);
    }
}
