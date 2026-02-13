<template>
  <div class="page-container">
    <!-- 筛选区 -->
    <div class="card-box">
      <div class="section-title">检测历史</div>
      <div class="filter-bar">
        <el-date-picker
          v-model="filter.dateRange"
          type="daterange"
          range-separator="至"
          start-placeholder="开始日期"
          end-placeholder="结束日期"
          @change="loadHistory"
        />
        <el-select v-model="filter.status" placeholder="状态" clearable @change="loadHistory">
          <el-option label="全部" value="" />
          <el-option label="处理中" value="processing" />
          <el-option label="已完成" value="completed" />
          <el-option label="失败" value="failed" />
        </el-select>
        <el-input
          v-model="filter.keyword"
          placeholder="搜索任务"
          style="width: 200px"
          clearable
          @keyup.enter="loadHistory"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
        <el-button @click="resetFilter">重置</el-button>
      </div>
    </div>

    <!-- 任务列表 -->
    <div class="card-box">
      <el-table :data="historyList" v-loading="loading" style="width: 100%">
        <el-table-column type="expand">
          <template #default="{ row }">
            <div class="expand-content" v-if="row.results">
              <div class="result-grid">
                <div 
                  v-for="result in row.results.slice(0, 4)" 
                  :key="result.id"
                  class="result-item"
                >
                  <el-image :src="result.overlayUrl || result.originalUrl" fit="cover" />
                  <div class="result-info">
                    <span>置信度: {{ (result.confidence * 100).toFixed(1) }}%</span>
                  </div>
                </div>
              </div>
              <el-button v-if="row.results.length > 4" type="primary" link @click="goDetail(row.id)">
                查看更多 (共{{ row.results.length }}张)
              </el-button>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="id" label="任务ID" width="120">
          <template #default="{ row }">
            <span class="job-id">#{{ row.id.toString().slice(-6) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="type" label="类型" width="100">
          <template #default="{ row }">
            <el-tag :type="row.type === 'batch' ? 'warning' : 'primary'" size="small">
              {{ row.type === 'batch' ? '批量' : '单张' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="imageCount" label="图像数" width="100" align="center" />
        <el-table-column prop="status" label="状态" width="120">
          <template #default="{ row }">
            <el-tag :type="getStatusType(row.status)">
              <el-icon v-if="row.status === 'processing'" class="is-loading">
                <Loading />
              </el-icon>
              {{ getStatusText(row.status) }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="progress" label="进度" width="150">
          <template #default="{ row }">
            <el-progress 
              :percentage="row.progress || 0" 
              :status="row.status === 'failed' ? 'exception' : ''"
              :stroke-width="8"
            />
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="创建时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column prop="duration" label="耗时" width="100">
          <template #default="{ row }">
            {{ row.duration ? `${row.duration}s` : '-' }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="180" fixed="right">
          <template #default="{ row }">
            <el-button 
              type="primary" 
              link 
              @click="goDetail(row.id)"
              :disabled="row.status === 'processing'"
            >
              查看结果
            </el-button>
            <el-button 
              v-if="row.status === 'failed'" 
              type="warning" 
              link 
              @click="retryJob(row)"
            >
              重试
            </el-button>
            <el-popconfirm
              title="确定要删除这个任务吗？"
              @confirm="deleteJob(row.id)"
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
          layout="total, sizes, prev, pager, next, jumper"
          @size-change="loadHistory"
          @current-change="loadHistory"
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import dayjs from 'dayjs'
import { getJobList, getJobStatus, deleteJob as apiDeleteJob } from '@/api/inference'

const router = useRouter()

const loading = ref(false)
const historyList = ref([])
const pagination = reactive({
  page: 1,
  size: 10,
  total: 0
})

const filter = reactive({
  dateRange: null,
  status: '',
  keyword: ''
})

let refreshTimer = null

// 获取状态类型
const getStatusType = (status) => {
  const types = {
    processing: 'warning',
    completed: 'success',
    failed: 'danger',
    pending: 'info'
  }
  return types[status] || 'info'
}

// 获取状态文本
const getStatusText = (status) => {
  const texts = {
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
    pending: '等待中'
  }
  return texts[status] || status
}

// 格式化时间
const formatTime = (time) => {
  return dayjs(time).format('YYYY-MM-DD HH:mm:ss')
}

// 加载历史记录
const loadHistory = async () => {
  try {
    loading.value = true
    const params = {
      page: pagination.page,
      size: pagination.size,
      status: filter.status,
      keyword: filter.keyword
    }
    
    if (filter.dateRange) {
      params.startDate = dayjs(filter.dateRange[0]).format('YYYY-MM-DD')
      params.endDate = dayjs(filter.dateRange[1]).format('YYYY-MM-DD')
    }

    const res = await getJobList(params)
    historyList.value = res.data?.records || []
    pagination.total = res.data?.total || 0

    // 如果有处理中的任务，启动自动刷新
    const hasProcessing = historyList.value.some(job => job.status === 'processing')
    if (hasProcessing && !refreshTimer) {
      startAutoRefresh()
    } else if (!hasProcessing && refreshTimer) {
      stopAutoRefresh()
    }
  } catch (error) {
    console.error('加载历史失败:', error)
  } finally {
    loading.value = false
  }
}

// 重置筛选
const resetFilter = () => {
  filter.dateRange = null
  filter.status = ''
  filter.keyword = ''
  pagination.page = 1
  loadHistory()
}

// 查看详情
const goDetail = (id) => {
  router.push(`/detection/${id}`)
}

// 重试任务
const retryJob = async (job) => {
  try {
    // 这里可以调用重试接口
    ElMessage.info('重试功能开发中')
  } catch (error) {
    console.error('重试失败:', error)
  }
}

// 删除任务
const deleteJob = async (id) => {
  try {
    await apiDeleteJob(id)
    ElMessage.success('删除成功')
    loadHistory()
  } catch (error) {
    console.error('删除失败:', error)
  }
}

// 自动刷新
const startAutoRefresh = () => {
  refreshTimer = setInterval(() => {
    loadHistory()
  }, 5000)
}

const stopAutoRefresh = () => {
  if (refreshTimer) {
    clearInterval(refreshTimer)
    refreshTimer = null
  }
}

onMounted(() => {
  loadHistory()
})

onUnmounted(() => {
  stopAutoRefresh()
})
</script>

<style lang="scss" scoped>
.filter-bar {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}

.job-id {
  font-family: monospace;
  color: #909399;
}

.expand-content {
  padding: 16px 16px 16px 60px;

  .result-grid {
    display: grid;
    grid-template-columns: repeat(4, 150px);
    gap: 12px;
    margin-bottom: 12px;
  }

  .result-item {
    border-radius: 6px;
    overflow: hidden;
    background: #f5f7fa;

    .el-image {
      width: 150px;
      height: 100px;
    }

    .result-info {
      padding: 8px;
      font-size: 12px;
      color: #606266;
    }
  }
}

.pagination-wrapper {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.is-loading {
  animation: rotating 2s linear infinite;
}

@keyframes rotating {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}
</style>
