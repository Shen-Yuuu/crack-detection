<template>
  <div class="page-container">
    <!-- 生成报告 -->
    <div class="card-box">
      <div class="section-title">生成报告</div>
      <el-form :model="generateForm" label-width="100px" class="generate-form">
        <el-row :gutter="20">
          <el-col :span="8">
            <el-form-item label="报告名称">
              <el-input v-model="generateForm.name" placeholder="请输入报告名称" />
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="检测任务">
              <el-select v-model="generateForm.jobId" placeholder="选择检测任务" filterable>
                <el-option 
                  v-for="job in completedJobs" 
                  :key="job.id" 
                  :label="`#${job.id.toString().slice(-6)} - ${formatTime(job.createdAt)}`"
                  :value="job.id" 
                />
              </el-select>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item label="报告格式">
              <el-radio-group v-model="generateForm.format">
                <el-radio value="pdf">PDF</el-radio>
                <el-radio value="excel">Excel</el-radio>
              </el-radio-group>
            </el-form-item>
          </el-col>
        </el-row>
        <el-row :gutter="20">
          <el-col :span="16">
            <el-form-item label="报告内容">
              <el-checkbox-group v-model="generateForm.sections">
                <el-checkbox value="summary">检测概览</el-checkbox>
                <el-checkbox value="statistics">统计分析</el-checkbox>
                <el-checkbox value="images">检测图像</el-checkbox>
                <el-checkbox value="details">详细数据</el-checkbox>
              </el-checkbox-group>
            </el-form-item>
          </el-col>
          <el-col :span="8">
            <el-form-item>
              <el-button type="primary" :loading="generating" @click="handleGenerate">
                <el-icon><Document /></el-icon>生成报告
              </el-button>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
    </div>

    <!-- 报告列表 -->
    <div class="card-box">
      <div class="section-title">历史报告</div>
      <el-table :data="reportList" v-loading="loading" style="width: 100%">
        <el-table-column prop="name" label="报告名称" min-width="200">
          <template #default="{ row }">
            <div class="report-name">
              <el-icon v-if="row.format === 'pdf'" color="#f56c6c"><Document /></el-icon>
              <el-icon v-else color="#67c23a"><Grid /></el-icon>
              <span>{{ row.name }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="format" label="格式" width="100" align="center">
          <template #default="{ row }">
            <el-tag :type="row.format === 'pdf' ? 'danger' : 'success'" size="small">
              {{ row.format.toUpperCase() }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="fileSize" label="大小" width="100">
          <template #default="{ row }">
            {{ formatFileSize(row.fileSize) }}
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="生成时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button 
              type="primary" 
              link 
              @click="previewReport(row)"
              :disabled="row.status !== 'completed'"
            >
              预览
            </el-button>
            <el-button 
              type="success" 
              link 
              @click="downloadReport(row)"
              :disabled="row.status !== 'completed'"
            >
              下载
            </el-button>
            <el-popconfirm
              title="确定要删除这个报告吗？"
              @confirm="deleteReport(row.id)"
            >
              <template #reference>
                <el-button type="danger" link>删除</el-button>
              </template>
            </el-popconfirm>
          </template>
        </el-table-column>
      </el-table>

      <!-- 分页 -->
      <div class="pagination-wrapper">
        <el-pagination
          v-model:current-page="pagination.page"
          v-model:page-size="pagination.size"
          :total="pagination.total"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          @size-change="loadReports"
          @current-change="loadReports"
        />
      </div>
    </div>

    <!-- PDF预览 -->
    <el-dialog v-model="previewVisible" title="报告预览" width="80%" top="5vh">
      <iframe 
        v-if="previewUrl && currentPreviewFormat === 'pdf'"
        :src="previewUrl" 
        class="preview-frame"
      />
      <div v-else class="preview-excel">
        <el-empty description="Excel 文件请下载后查看" />
      </div>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import dayjs from 'dayjs'
import { getJobList } from '@/api/inference'
import { 
  getReportList, 
  generateReport, 
  downloadReport as apiDownloadReport,
  deleteReport as apiDeleteReport 
} from '@/api/report'

const loading = ref(false)
const generating = ref(false)
const reportList = ref([])
const completedJobs = ref([])

const pagination = reactive({
  page: 1,
  size: 10,
  total: 0
})

const generateForm = reactive({
  name: '',
  jobId: null,
  format: 'pdf',
  sections: ['summary', 'statistics', 'images']
})

// 预览相关
const previewVisible = ref(false)
const previewUrl = ref('')
const currentPreviewFormat = ref('')

// 获取状态类型
const getStatusType = (status) => {
  const types = {
    generating: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

// 获取状态文本
const getStatusText = (status) => {
  const texts = {
    generating: '生成中',
    completed: '已完成',
    failed: '生成失败'
  }
  return texts[status] || status
}

// 格式化时间
const formatTime = (time) => {
  return dayjs(time).format('YYYY-MM-DD HH:mm')
}

// 格式化文件大小
const formatFileSize = (bytes) => {
  if (!bytes) return '-'
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB'
}

// 加载已完成的检测任务
const loadCompletedJobs = async () => {
  try {
    const res = await getJobList({ status: 'completed', size: 100 })
    completedJobs.value = res.data?.records || []
  } catch (error) {
    console.error('加载检测任务失败:', error)
  }
}

// 加载报告列表
const loadReports = async () => {
  try {
    loading.value = true
    const res = await getReportList({
      page: pagination.page,
      size: pagination.size
    })
    reportList.value = res.data?.records || []
    pagination.total = res.data?.total || 0
  } catch (error) {
    console.error('加载报告失败:', error)
  } finally {
    loading.value = false
  }
}

// 生成报告
const handleGenerate = async () => {
  if (!generateForm.name) {
    ElMessage.warning('请输入报告名称')
    return
  }
  if (!generateForm.jobId) {
    ElMessage.warning('请选择检测任务')
    return
  }
  if (generateForm.sections.length === 0) {
    ElMessage.warning('请选择报告内容')
    return
  }

  try {
    generating.value = true
    await generateReport({
      name: generateForm.name,
      jobId: generateForm.jobId,
      format: generateForm.format,
      sections: generateForm.sections
    })
    ElMessage.success('报告生成任务已提交')
    generateForm.name = ''
    generateForm.jobId = null
    loadReports()
  } catch (error) {
    console.error('生成报告失败:', error)
  } finally {
    generating.value = false
  }
}

// 预览报告
const previewReport = (report) => {
  currentPreviewFormat.value = report.format
  if (report.format === 'pdf') {
    previewUrl.value = report.fileUrl
    previewVisible.value = true
  } else {
    // Excel直接下载
    downloadReport(report)
  }
}

// 下载报告
const downloadReport = async (report) => {
  try {
    const res = await apiDownloadReport(report.id)
    // 创建下载链接
    const blob = new Blob([res.data])
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `${report.name}.${report.format}`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    window.URL.revokeObjectURL(url)
  } catch (error) {
    console.error('下载失败:', error)
    // 备用方案：直接打开URL
    if (report.fileUrl) {
      window.open(report.fileUrl, '_blank')
    }
  }
}

// 删除报告
const deleteReport = async (id) => {
  try {
    await apiDeleteReport(id)
    ElMessage.success('删除成功')
    loadReports()
  } catch (error) {
    console.error('删除失败:', error)
  }
}

onMounted(() => {
  loadCompletedJobs()
  loadReports()
})
</script>

<style lang="scss" scoped>
.generate-form {
  .el-form-item {
    margin-bottom: 18px;
  }
}

.report-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.pagination-wrapper {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.preview-frame {
  width: 100%;
  height: 70vh;
  border: none;
}

.preview-excel {
  height: 300px;
  display: flex;
  align-items: center;
  justify-content: center;
}
</style>
