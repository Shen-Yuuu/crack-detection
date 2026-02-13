import request from '@/utils/request'

// 生成叠加图
export function generateOverlay(resultId, config) {
  return request({
    url: `/visual/overlay/${resultId}`,
    method: 'post',
    data: config
  })
}

// 生成热力图
export function generateHeatmap(resultId, config) {
  return request({
    url: `/visual/heatmap/${resultId}`,
    method: 'post',
    data: config
  })
}

// 获取统计信息
export function getStatistics(resultId) {
  return request({
    url: `/visual/statistics/${resultId}`,
    method: 'get'
  })
}
