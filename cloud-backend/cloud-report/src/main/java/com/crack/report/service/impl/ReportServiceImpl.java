package com.crack.report.service.impl;

import com.alibaba.excel.EasyExcel;
import com.alibaba.fastjson2.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.crack.common.entity.DetectionResult;
import com.crack.common.entity.Report;
import com.crack.common.exception.BusinessException;
import com.crack.common.result.PageResult;
import com.crack.common.result.ResultCode;
import com.crack.common.utils.MinioService;
import com.crack.report.dto.GenerateReportRequest;
import com.crack.report.dto.ReportResponse;
import com.crack.report.mapper.DetectionResultMapper;
import com.crack.report.mapper.ReportMapper;
import com.crack.report.service.ReportService;
import com.itextpdf.text.*;
import com.itextpdf.text.pdf.BaseFont;
import com.itextpdf.text.pdf.PdfPCell;
import com.itextpdf.text.pdf.PdfPTable;
import com.itextpdf.text.pdf.PdfWriter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;

/**
 * 报告服务实现
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ReportServiceImpl implements ReportService {

    private final ReportMapper reportMapper;
    private final DetectionResultMapper resultMapper;
    private final MinioService minioService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ReportResponse generateReport(GenerateReportRequest request, Long userId) {
        log.info("生成报告: {}, 用户ID: {}", request.getTitle(), userId);

        // 创建报告记录
        Report report = new Report();
        report.setUserId(userId);
        report.setTitle(request.getTitle());
        report.setReportType(request.getReportType());
        report.setResultIds(JSON.toJSONString(request.getResultIds()));
        report.setStatus("generating");

        reportMapper.insert(report);

        // 异步生成报告
        generateReportAsync(report, request);

        return convertToResponse(report);
    }

    @Async
    protected void generateReportAsync(Report report, GenerateReportRequest request) {
        try {
            String fileUrl;

            if ("excel".equalsIgnoreCase(request.getReportType())) {
                fileUrl = generateExcelReport(report, request);
            } else {
                fileUrl = generatePdfReport(report, request);
            }

            // 更新报告状态
            report.setFileUrl(fileUrl);
            report.setStatus("completed");
            reportMapper.updateById(report);

            log.info("报告生成完成: {}", report.getId());

        } catch (Exception e) {
            log.error("报告生成失败", e);
            report.setStatus("failed");
            reportMapper.updateById(report);
        }
    }

    /**
     * 生成PDF报告
     */
    private String generatePdfReport(Report report, GenerateReportRequest request) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        Document document = new Document(PageSize.A4);
        PdfWriter.getInstance(document, baos);
        document.open();

        // 添加标题
        Font titleFont = new Font(Font.FontFamily.HELVETICA, 24, Font.BOLD);
        Paragraph title = new Paragraph(request.getTitle(), titleFont);
        title.setAlignment(Element.ALIGN_CENTER);
        title.setSpacingAfter(20);
        document.add(title);

        // 添加生成时间
        Font normalFont = new Font(Font.FontFamily.HELVETICA, 12);
        Paragraph date = new Paragraph(
                "Generated: " + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")),
                normalFont
        );
        date.setAlignment(Element.ALIGN_RIGHT);
        date.setSpacingAfter(30);
        document.add(date);

        // 获取检测结果
        List<DetectionResult> results = resultMapper.selectBatchIds(request.getResultIds());

        // 添加统计摘要
        Paragraph summary = new Paragraph("Detection Summary", new Font(Font.FontFamily.HELVETICA, 16, Font.BOLD));
        summary.setSpacingAfter(10);
        document.add(summary);

        // 创建表格
        PdfPTable table = new PdfPTable(5);
        table.setWidthPercentage(100);

        // 表头
        String[] headers = {"ID", "Confidence", "Crack Count", "Total Area", "Processing Time"};
        for (String header : headers) {
            PdfPCell cell = new PdfPCell(new Phrase(header, new Font(Font.FontFamily.HELVETICA, 10, Font.BOLD)));
            cell.setHorizontalAlignment(Element.ALIGN_CENTER);
            cell.setBackgroundColor(BaseColor.LIGHT_GRAY);
            cell.setPadding(5);
            table.addCell(cell);
        }

        // 数据行
        for (DetectionResult result : results) {
            table.addCell(String.valueOf(result.getId()));
            table.addCell(result.getConfidence() != null ? result.getConfidence().toString() : "N/A");
            table.addCell(result.getCrackCount() != null ? result.getCrackCount().toString() : "N/A");
            table.addCell(result.getTotalArea() != null ? result.getTotalArea().toString() : "N/A");
            table.addCell(result.getProcessingTime() != null ? result.getProcessingTime().toString() + "s" : "N/A");
        }

        document.add(table);

        document.close();

        // 上传到MinIO
        byte[] bytes = baos.toByteArray();
        String filePath = "reports/" + UUID.randomUUID() + ".pdf";
        minioService.uploadStream(
                new ByteArrayInputStream(bytes),
                filePath,
                "application/pdf",
                bytes.length
        );

        return filePath;
    }

    /**
     * 生成Excel报告
     */
    private String generateExcelReport(Report report, GenerateReportRequest request) throws Exception {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        // 获取检测结果
        List<DetectionResult> results = resultMapper.selectBatchIds(request.getResultIds());

        // 转换为Excel数据
        List<List<Object>> data = new ArrayList<>();

        // 表头
        List<Object> headers = List.of("ID", "任务ID", "置信度", "裂纹数量", "总面积", "处理时间", "创建时间");
        data.add(headers);

        // 数据行
        for (DetectionResult result : results) {
            List<Object> row = new ArrayList<>();
            row.add(result.getId());
            row.add(result.getJobId());
            row.add(result.getConfidence() != null ? result.getConfidence().toString() : "");
            row.add(result.getCrackCount());
            row.add(result.getTotalArea() != null ? result.getTotalArea().toString() : "");
            row.add(result.getProcessingTime() != null ? result.getProcessingTime().toString() : "");
            row.add(result.getCreatedAt() != null ? result.getCreatedAt().toString() : "");
            data.add(row);
        }

        EasyExcel.write(baos)
                .sheet("检测结果")
                .doWrite(data);

        // 上传到MinIO
        byte[] bytes = baos.toByteArray();
        String filePath = "reports/" + UUID.randomUUID() + ".xlsx";
        minioService.uploadStream(
                new ByteArrayInputStream(bytes),
                filePath,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                bytes.length
        );

        return filePath;
    }

    @Override
    public ReportResponse getReport(Long reportId, Long userId) {
        Report report = getReportByIdAndUser(reportId, userId);
        return convertToResponse(report);
    }

    @Override
    public PageResult<ReportResponse> listReports(Integer page, Integer size, Long userId) {
        Page<Report> pageObj = new Page<>(page, size);

        LambdaQueryWrapper<Report> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Report::getUserId, userId);
        wrapper.orderByDesc(Report::getCreatedAt);

        Page<Report> resultPage = reportMapper.selectPage(pageObj, wrapper);

        List<ReportResponse> records = resultPage.getRecords().stream()
                .map(this::convertToResponse)
                .collect(Collectors.toList());

        return PageResult.of(
                resultPage.getCurrent(),
                resultPage.getSize(),
                resultPage.getTotal(),
                records
        );
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteReport(Long reportId, Long userId) {
        log.info("删除报告: {}", reportId);

        Report report = getReportByIdAndUser(reportId, userId);

        // 删除文件
        if (report.getFileUrl() != null) {
            try {
                minioService.deleteObject(report.getFileUrl());
            } catch (Exception e) {
                log.warn("删除报告文件失败: {}", e.getMessage());
            }
        }

        reportMapper.deleteById(reportId);
        log.info("报告删除成功: {}", reportId);
    }

    @Override
    public String getDownloadUrl(Long reportId, Long userId) {
        Report report = getReportByIdAndUser(reportId, userId);

        if (!"completed".equals(report.getStatus())) {
            throw new BusinessException("报告尚未生成完成");
        }

        return minioService.getPresignedUrl(report.getFileUrl());
    }

    /**
     * 根据ID和用户获取报告
     */
    private Report getReportByIdAndUser(Long reportId, Long userId) {
        Report report = reportMapper.selectOne(
                new LambdaQueryWrapper<Report>()
                        .eq(Report::getId, reportId)
                        .eq(Report::getUserId, userId)
        );
        if (report == null) {
            throw new BusinessException(ResultCode.REPORT_NOT_FOUND);
        }
        return report;
    }

    /**
     * 转换为响应对象
     */
    private ReportResponse convertToResponse(Report report) {
        return ReportResponse.builder()
                .id(report.getId())
                .title(report.getTitle())
                .reportType(report.getReportType())
                .fileUrl(report.getFileUrl() != null ? minioService.getPresignedUrl(report.getFileUrl()) : null)
                .status(report.getStatus())
                .createdAt(report.getCreatedAt())
                .build();
    }
}
