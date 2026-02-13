package com.crack.inference.service.impl;

import cn.hutool.core.io.IoUtil;
import com.alibaba.fastjson2.JSON;
import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.crack.common.constant.Constants;
import com.crack.common.entity.DetectionJob;
import com.crack.common.entity.DetectionResult;
import com.crack.common.exception.BusinessException;
import com.crack.common.result.PageResult;
import com.crack.common.result.ResultCode;
import com.crack.common.utils.MinioService;
import com.crack.common.utils.RedisService;
import com.crack.inference.client.PythonInferenceClient;
import com.crack.inference.dto.*;
import com.crack.inference.mapper.DetectionJobMapper;
import com.crack.inference.mapper.DetectionResultMapper;
import com.crack.inference.service.InferenceService;
import com.crack.inference.utils.ByteArrayMultipartFile;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * 推理服务实现
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class InferenceServiceImpl implements InferenceService {

    private final DetectionJobMapper jobMapper;
    private final DetectionResultMapper resultMapper;
    private final PythonInferenceClient pythonClient;
    private final MinioService minioService;
    private final RedisService redisService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public DetectionJobResponse createJob(MultipartFile file, String modelVersion,
                                          InferenceConfig config, Long userId) {
        log.info("创建推理任务，用户ID: {}, 文件名: {}", userId, file.getOriginalFilename());

        // 1. 上传图像到MinIO
        String imageUrl = minioService.uploadFile(file, Constants.FOLDER_IMAGES);

        // 2. 创建任务记录
        DetectionJob job = new DetectionJob();
        job.setUserId(userId);
        job.setImageUrl(imageUrl);
        job.setImageName(file.getOriginalFilename());
        job.setModelVersion(modelVersion != null ? modelVersion : "v1.0");
        job.setConfig(JSON.toJSONString(config != null ? config : new InferenceConfig()));
        job.setStatus(Constants.JOB_STATUS_PENDING);
        jobMapper.insert(job);

        // 3. 异步执行推理
        executeInferenceAsync(job, file, config);

        log.info("推理任务创建成功，任务ID: {}", job.getId());
        return convertToJobResponse(job);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public DetectionJobResponse createJobByUrl(String imageUrl, String modelVersion,
                                               InferenceConfig config, Long userId) {
        log.info("通过URL创建推理任务，用户ID: {}, URL: {}", userId, imageUrl);

        // 创建任务记录
        DetectionJob job = new DetectionJob();
        job.setUserId(userId);
        job.setImageUrl(imageUrl);
        job.setImageName(extractFileName(imageUrl));
        job.setModelVersion(modelVersion != null ? modelVersion : "v1.0");
        job.setConfig(JSON.toJSONString(config != null ? config : new InferenceConfig()));
        job.setStatus(Constants.JOB_STATUS_PENDING);
        jobMapper.insert(job);

        // 异步执行推理
        executeInferenceByUrlAsync(job, config);

        return convertToJobResponse(job);
    }

    @Override
    public BatchInferenceResponse batchCreateJobs(BatchInferenceRequest request, Long userId) {
        log.info("批量创建推理任务，用户ID: {}, 数量: {}", userId, request.getImageUrls().size());

        List<Long> jobIds = new ArrayList<>();
        for (String imageUrl : request.getImageUrls()) {
            DetectionJobResponse response = createJobByUrl(
                    imageUrl, 
                    request.getModelVersion(), 
                    request.getConfig(), 
                    userId
            );
            jobIds.add(response.getJobId());
        }

        return BatchInferenceResponse.builder()
                .batchId(UUID.randomUUID().toString())
                .totalJobs(jobIds.size())
                .jobIds(jobIds)
                .status("submitted")
                .build();
    }

    @Override
    public DetectionJobResponse getJobStatus(Long jobId, Long userId) {
        DetectionJob job = getJobByIdAndUser(jobId, userId);
        return convertToJobResponse(job);
    }

    @Override
    public DetectionResultResponse getResult(Long jobId, Long userId) {
        // 先从缓存获取
        String cacheKey = Constants.REDIS_INFERENCE_RESULT_PREFIX + jobId;
        String cachedResult = redisService.get(cacheKey);
        if (cachedResult != null) {
            return JSON.parseObject(cachedResult, DetectionResultResponse.class);
        }

        // 从数据库获取
        DetectionJob job = getJobByIdAndUser(jobId, userId);

        if (!Constants.JOB_STATUS_COMPLETED.equals(job.getStatus())) {
            return DetectionResultResponse.builder()
                    .jobId(jobId)
                    .status(job.getStatus())
                    .build();
        }

        DetectionResult result = resultMapper.selectOne(
                new LambdaQueryWrapper<DetectionResult>()
                        .eq(DetectionResult::getJobId, jobId)
        );

        if (result == null) {
            throw new BusinessException(ResultCode.JOB_NOT_FOUND, "检测结果不存在");
        }

        DetectionResultResponse response = convertToResultResponse(job, result);

        // 缓存结果
        redisService.setEx(cacheKey, JSON.toJSONString(response), 3600);

        return response;
    }

    @Override
    public PageResult<DetectionJobResponse> listJobs(JobQueryParam param, Long userId) {
        Page<DetectionJob> page = new Page<>(param.getPage(), param.getSize());

        LambdaQueryWrapper<DetectionJob> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(DetectionJob::getUserId, userId);

        if (StringUtils.hasText(param.getStatus())) {
            wrapper.eq(DetectionJob::getStatus, param.getStatus());
        }
        if (StringUtils.hasText(param.getModelVersion())) {
            wrapper.eq(DetectionJob::getModelVersion, param.getModelVersion());
        }

        wrapper.orderByDesc(DetectionJob::getCreatedAt);

        Page<DetectionJob> resultPage = jobMapper.selectPage(page, wrapper);

        List<DetectionJobResponse> records = resultPage.getRecords().stream()
                .map(this::convertToJobResponse)
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
    public void deleteJob(Long jobId, Long userId) {
        log.info("删除任务，任务ID: {}, 用户ID: {}", jobId, userId);

        DetectionJob job = getJobByIdAndUser(jobId, userId);

        // 删除MinIO中的文件
        if (StringUtils.hasText(job.getImageUrl())) {
            try {
                minioService.deleteObject(job.getImageUrl());
            } catch (Exception e) {
                log.warn("删除图像文件失败: {}", e.getMessage());
            }
        }

        // 删除结果
        resultMapper.delete(new LambdaQueryWrapper<DetectionResult>()
                .eq(DetectionResult::getJobId, jobId));

        // 删除任务
        jobMapper.deleteById(jobId);

        // 删除缓存
        String cacheKey = Constants.REDIS_INFERENCE_RESULT_PREFIX + jobId;
        redisService.delete(cacheKey);

        log.info("任务删除成功，任务ID: {}", jobId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public DetectionJobResponse retryJob(Long jobId, Long userId) {
        log.info("重新执行任务，任务ID: {}, 用户ID: {}", jobId, userId);

        DetectionJob job = getJobByIdAndUser(jobId, userId);

        if (Constants.JOB_STATUS_RUNNING.equals(job.getStatus())) {
            throw new BusinessException(ResultCode.JOB_STILL_RUNNING);
        }

        // 删除旧的结果
        resultMapper.delete(new LambdaQueryWrapper<DetectionResult>()
                .eq(DetectionResult::getJobId, jobId));

        // 更新任务状态
        job.setStatus(Constants.JOB_STATUS_PENDING);
        job.setErrorMessage(null);
        job.setResultId(null);
        jobMapper.updateById(job);

        // 重新执行推理
        InferenceConfig config = JSON.parseObject(job.getConfig(), InferenceConfig.class);
        executeInferenceByUrlAsync(job, config);

        return convertToJobResponse(job);
    }

    /**
     * 异步执行推理（上传的文件）
     */
    @Async
    protected void executeInferenceAsync(DetectionJob job, MultipartFile file, InferenceConfig config) {
        try {
            // 更新状态为运行中
            updateJobStatus(job.getId(), Constants.JOB_STATUS_RUNNING, null);

            // 调用Python推理服务
            PythonInferenceResponse pythonResponse = pythonClient.detect(
                    file,
                    config != null ? config.getThreshold().floatValue() : 0.5f,
                    config != null && config.getUseTta(),
                    config == null || config.getReturnMask(),
                    config == null || config.getReturnOverlay()
            );

            // 保存结果
            saveInferenceResult(job, pythonResponse);

        } catch (Exception e) {
            log.error("推理执行失败，任务ID: {}", job.getId(), e);
            updateJobStatus(job.getId(), Constants.JOB_STATUS_FAILED, e.getMessage());
        }
    }

    /**
     * 异步执行推理（URL）
     */
    @Async
    protected void executeInferenceByUrlAsync(DetectionJob job, InferenceConfig config) {
        try {
            // 更新状态为运行中
            updateJobStatus(job.getId(), Constants.JOB_STATUS_RUNNING, null);

            // 从MinIO下载图像
            InputStream inputStream = minioService.getObject(job.getImageUrl());
            byte[] bytes = IoUtil.readBytes(inputStream);

            // 构造MultipartFile
            MultipartFile file = new ByteArrayMultipartFile(
                    "file",
                    job.getImageName(),
                    "image/jpeg",
                    bytes
            );

            // 调用Python推理服务
            PythonInferenceResponse pythonResponse = pythonClient.detect(
                    file,
                    config != null ? config.getThreshold().floatValue() : 0.5f,
                    config != null && config.getUseTta(),
                    config == null || config.getReturnMask(),
                    config == null || config.getReturnOverlay()
            );

            // 保存结果
            saveInferenceResult(job, pythonResponse);

        } catch (Exception e) {
            log.error("推理执行失败，任务ID: {}", job.getId(), e);
            updateJobStatus(job.getId(), Constants.JOB_STATUS_FAILED, e.getMessage());
        }
    }

    /**
     * 保存推理结果
     */
    private void saveInferenceResult(DetectionJob job, PythonInferenceResponse pythonResponse) {
        // 保存掩码图到MinIO
        String maskUrl = null;
        if (pythonResponse.getMaskBase64() != null) {
            byte[] maskBytes = Base64.getDecoder().decode(pythonResponse.getMaskBase64());
            String maskName = Constants.FOLDER_MASKS + "/" + UUID.randomUUID() + ".png";
            minioService.uploadStream(
                    new ByteArrayInputStream(maskBytes),
                    maskName,
                    "image/png",
                    maskBytes.length
            );
            maskUrl = maskName;
        }

        // 保存叠加图到MinIO
        String overlayUrl = null;
        if (pythonResponse.getOverlayBase64() != null) {
            byte[] overlayBytes = Base64.getDecoder().decode(pythonResponse.getOverlayBase64());
            String overlayName = Constants.FOLDER_OVERLAYS + "/" + UUID.randomUUID() + ".png";
            minioService.uploadStream(
                    new ByteArrayInputStream(overlayBytes),
                    overlayName,
                    "image/png",
                    overlayBytes.length
            );
            overlayUrl = overlayName;
        }

        // 创建结果记录
        DetectionResult result = new DetectionResult();
        result.setJobId(job.getId());
        result.setMaskUrl(maskUrl);
        result.setOverlayUrl(overlayUrl);
        result.setConfidence(pythonResponse.getConfidence());
        result.setProcessingTime(pythonResponse.getProcessingTime());
        result.setCrackCount(pythonResponse.getCrackCount());
        result.setTotalArea(pythonResponse.getTotalArea());
        resultMapper.insert(result);

        // 更新任务状态
        job.setResultId(result.getId());
        job.setStatus(Constants.JOB_STATUS_COMPLETED);
        jobMapper.updateById(job);

        log.info("推理完成，任务ID: {}, 结果ID: {}", job.getId(), result.getId());
    }

    /**
     * 更新任务状态
     */
    private void updateJobStatus(Long jobId, String status, String errorMessage) {
        DetectionJob job = new DetectionJob();
        job.setId(jobId);
        job.setStatus(status);
        job.setErrorMessage(errorMessage);
        jobMapper.updateById(job);
    }

    /**
     * 根据ID和用户获取任务
     */
    private DetectionJob getJobByIdAndUser(Long jobId, Long userId) {
        DetectionJob job = jobMapper.selectOne(
                new LambdaQueryWrapper<DetectionJob>()
                        .eq(DetectionJob::getId, jobId)
                        .eq(DetectionJob::getUserId, userId)
        );
        if (job == null) {
            throw new BusinessException(ResultCode.JOB_NOT_FOUND);
        }
        return job;
    }

    /**
     * 从URL中提取文件名
     */
    private String extractFileName(String url) {
        if (url == null) return "unknown";
        int lastSlash = url.lastIndexOf('/');
        if (lastSlash >= 0 && lastSlash < url.length() - 1) {
            return url.substring(lastSlash + 1);
        }
        return "unknown";
    }

    /**
     * 转换为任务响应
     */
    private DetectionJobResponse convertToJobResponse(DetectionJob job) {
        return DetectionJobResponse.builder()
                .jobId(job.getId())
                .status(job.getStatus())
                .imageUrl(minioService.getPresignedUrl(job.getImageUrl()))
                .modelVersion(job.getModelVersion())
                .createdAt(job.getCreatedAt())
                .errorMessage(job.getErrorMessage())
                .build();
    }

    /**
     * 转换为结果响应
     */
    private DetectionResultResponse convertToResultResponse(DetectionJob job, DetectionResult result) {
        return DetectionResultResponse.builder()
                .resultId(result.getId())
                .jobId(job.getId())
                .status(job.getStatus())
                .imageUrl(minioService.getPresignedUrl(job.getImageUrl()))
                .maskUrl(result.getMaskUrl() != null ? minioService.getPresignedUrl(result.getMaskUrl()) : null)
                .overlayUrl(result.getOverlayUrl() != null ? minioService.getPresignedUrl(result.getOverlayUrl()) : null)
                .heatmapUrl(result.getHeatmapUrl() != null ? minioService.getPresignedUrl(result.getHeatmapUrl()) : null)
                .confidence(result.getConfidence())
                .crackCount(result.getCrackCount())
                .totalArea(result.getTotalArea())
                .processingTime(result.getProcessingTime())
                .createdAt(result.getCreatedAt())
                .build();
    }
}
