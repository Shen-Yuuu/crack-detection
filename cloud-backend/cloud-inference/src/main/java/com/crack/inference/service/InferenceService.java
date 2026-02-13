package com.crack.inference.service;

import com.crack.common.result.PageResult;
import com.crack.inference.dto.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * 推理服务接口
 */
public interface InferenceService {

    /**
     * 创建推理任务（上传图像）
     *
     * @param file         图像文件
     * @param modelVersion 模型版本
     * @param config       推理配置
     * @param userId       用户ID
     * @return 任务响应
     */
    DetectionJobResponse createJob(MultipartFile file, String modelVersion, 
                                   InferenceConfig config, Long userId);

    /**
     * 通过URL创建推理任务
     *
     * @param imageUrl     图像URL
     * @param modelVersion 模型版本
     * @param config       推理配置
     * @param userId       用户ID
     * @return 任务响应
     */
    DetectionJobResponse createJobByUrl(String imageUrl, String modelVersion,
                                        InferenceConfig config, Long userId);

    /**
     * 批量创建推理任务
     *
     * @param request 批量请求
     * @param userId  用户ID
     * @return 批量响应
     */
    BatchInferenceResponse batchCreateJobs(BatchInferenceRequest request, Long userId);

    /**
     * 获取任务状态
     *
     * @param jobId  任务ID
     * @param userId 用户ID
     * @return 任务响应
     */
    DetectionJobResponse getJobStatus(Long jobId, Long userId);

    /**
     * 获取检测结果
     *
     * @param jobId  任务ID
     * @param userId 用户ID
     * @return 结果响应
     */
    DetectionResultResponse getResult(Long jobId, Long userId);

    /**
     * 获取任务列表
     *
     * @param param  查询参数
     * @param userId 用户ID
     * @return 分页结果
     */
    PageResult<DetectionJobResponse> listJobs(JobQueryParam param, Long userId);

    /**
     * 删除任务
     *
     * @param jobId  任务ID
     * @param userId 用户ID
     */
    void deleteJob(Long jobId, Long userId);

    /**
     * 重新执行任务
     *
     * @param jobId  任务ID
     * @param userId 用户ID
     * @return 任务响应
     */
    DetectionJobResponse retryJob(Long jobId, Long userId);
}
