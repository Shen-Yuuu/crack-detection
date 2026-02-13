package com.crack.inference.controller;

import com.crack.common.result.PageResult;
import com.crack.common.result.Result;
import com.crack.inference.dto.*;
import com.crack.inference.service.InferenceService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * 推理控制器
 */
@RestController
@RequestMapping("/api/v1/inference")
@RequiredArgsConstructor
@Tag(name = "推理管理", description = "道路裂纹检测推理相关接口")
public class InferenceController {

    private final InferenceService inferenceService;

    @PostMapping("/detect")
    @Operation(summary = "单张图像检测", description = "上传图像进行裂纹检测")
    public Result<DetectionJobResponse> detect(
            @Parameter(description = "图像文件") @RequestPart("file") MultipartFile file,
            @Parameter(description = "模型版本") @RequestParam(value = "modelVersion", defaultValue = "v1.0") String modelVersion,
            @Parameter(description = "检测阈值") @RequestParam(value = "threshold", defaultValue = "0.5") Float threshold,
            @Parameter(description = "是否使用TTA") @RequestParam(value = "useTta", defaultValue = "false") Boolean useTta,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        InferenceConfig config = new InferenceConfig();
        config.setThreshold(new java.math.BigDecimal(threshold.toString()));
        config.setUseTta(useTta);

        DetectionJobResponse response = inferenceService.createJob(file, modelVersion, config, userId);
        return Result.success("任务创建成功", response);
    }

    @PostMapping("/detect/url")
    @Operation(summary = "通过URL检测", description = "通过图像URL进行裂纹检测")
    public Result<DetectionJobResponse> detectByUrl(
            @Parameter(description = "图像URL") @RequestParam("imageUrl") String imageUrl,
            @Parameter(description = "模型版本") @RequestParam(value = "modelVersion", defaultValue = "v1.0") String modelVersion,
            @RequestBody(required = false) InferenceConfig config,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        DetectionJobResponse response = inferenceService.createJobByUrl(imageUrl, modelVersion, config, userId);
        return Result.success("任务创建成功", response);
    }

    @PostMapping("/batch")
    @Operation(summary = "批量检测", description = "批量提交图像进行裂纹检测")
    public Result<BatchInferenceResponse> batchDetect(
            @RequestBody BatchInferenceRequest request,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        BatchInferenceResponse response = inferenceService.batchCreateJobs(request, userId);
        return Result.success("批量任务创建成功", response);
    }

    @GetMapping("/jobs")
    @Operation(summary = "任务列表", description = "获取检测任务列表")
    public Result<PageResult<DetectionJobResponse>> listJobs(
            JobQueryParam param,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        PageResult<DetectionJobResponse> result = inferenceService.listJobs(param, userId);
        return Result.success(result);
    }

    @GetMapping("/jobs/{jobId}")
    @Operation(summary = "任务状态", description = "获取检测任务状态")
    public Result<DetectionJobResponse> getJobStatus(
            @Parameter(description = "任务ID") @PathVariable Long jobId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        DetectionJobResponse response = inferenceService.getJobStatus(jobId, userId);
        return Result.success(response);
    }

    @GetMapping("/result/{jobId}")
    @Operation(summary = "检测结果", description = "获取检测结果详情")
    public Result<DetectionResultResponse> getResult(
            @Parameter(description = "任务ID") @PathVariable Long jobId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        DetectionResultResponse response = inferenceService.getResult(jobId, userId);
        return Result.success(response);
    }

    @DeleteMapping("/jobs/{jobId}")
    @Operation(summary = "删除任务", description = "删除检测任务及其结果")
    public Result<Void> deleteJob(
            @Parameter(description = "任务ID") @PathVariable Long jobId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        inferenceService.deleteJob(jobId, userId);
        return Result.success("任务删除成功", null);
    }

    @PostMapping("/jobs/{jobId}/retry")
    @Operation(summary = "重新执行", description = "重新执行失败的任务")
    public Result<DetectionJobResponse> retryJob(
            @Parameter(description = "任务ID") @PathVariable Long jobId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        DetectionJobResponse response = inferenceService.retryJob(jobId, userId);
        return Result.success("任务重新提交成功", response);
    }
}
