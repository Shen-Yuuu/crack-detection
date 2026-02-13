import request from '@/utils/request'

// 创建数据集
export function createDataset(data) {
  return request({
    url: '/dataset',
    method: 'post',
    data
  })
}

// 获取数据集列表
export function getDatasetList(params) {
  return request({
    url: '/dataset',
    method: 'get',
    params
  })
}

// 获取数据集详情
export function getDatasetDetail(id) {
  return request({
    url: `/dataset/${id}`,
    method: 'get'
  })
}

// 删除数据集
export function deleteDataset(id) {
  return request({
    url: `/dataset/${id}`,
    method: 'delete'
  })
}

// 上传图像
export function uploadImage(datasetId, formData, onProgress) {
  return request({
    url: `/dataset/${datasetId}/images`,
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    onUploadProgress: onProgress
  })
}

// 批量上传图像
export function batchUploadImages(datasetId, formData, onProgress) {
  return request({
    url: `/dataset/${datasetId}/images/batch`,
    method: 'post',
    data: formData,
    headers: {
      'Content-Type': 'multipart/form-data'
    },
    timeout: 300000, // 5分钟超时
    onUploadProgress: onProgress
  })
}

// 获取图像列表
export function getImageList(datasetId, params) {
  return request({
    url: `/dataset/${datasetId}/images`,
    method: 'get',
    params
  })
}

// 删除图像
export function deleteImage(datasetId, imageId) {
  return request({
    url: `/dataset/${datasetId}/images/${imageId}`,
    method: 'delete'
  })
}
