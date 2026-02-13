import request from '@/utils/request'

// 生成报告
export function generateReport(data) {
  return request({
    url: '/report/generate',
    method: 'post',
    data
  })
}

// 获取报告列表
export function getReportList(params) {
  return request({
    url: '/report',
    method: 'get',
    params
  })
}

// 获取报告详情
export function getReportDetail(id) {
  return request({
    url: `/report/${id}`,
    method: 'get'
  })
}

// 删除报告
export function deleteReport(id) {
  return request({
    url: `/report/${id}`,
    method: 'delete'
  })
}

// 下载报告
export function downloadReport(id) {
  return request({
    url: `/report/${id}/download`,
    method: 'get'
  })
}
