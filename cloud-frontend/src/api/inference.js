import request from '@/utils/request'

// 单张图像检测
export function detectImage(formData, onProgress) {
  return request({
    url: '/inference/detect',
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    timeout: 120000,
    onUploadProgress: onProgress
  })
}

// 批量检测
export function batchDetect(data) {
  return request({
    url: '/inference/batch',
    method: 'post',
    data
  })
}

// 获取检测任务状态
export function getJobStatus(jobId) {
  return request({
    url: `/inference/jobs/${jobId}`,
    method: 'get'
  })
}

// 获取检测结果
export function getDetectionResult(jobId) {
  return request({
    url: `/inference/result/${jobId}`,
    method: 'get'
  })
}

// 获取任务列表
export function getJobList(params) {
  return request({
    url: '/inference/jobs',
    method: 'get',
    params
  })
}

// 删除任务
export function deleteJob(jobId) {
  return request({
    url: `/inference/jobs/${jobId}`,
    method: 'delete'
  })
}

// 重试任务
export function retryJob(jobId) {
  return request({
    url: `/inference/jobs/${jobId}/retry`,
    method: 'post'
  })
}
