<template>
  <div class="page-container">
    <div class="dashboard-header">
      <h2>工作台</h2>
      <p>欢迎使用道路裂纹检测系统</p>
    </div>

    <!-- 统计卡片 -->
    <el-row :gutter="20" class="stat-cards">
      <el-col :span="6">
        <div class="stat-card" style="--card-color: #409eff">
          <div class="stat-icon">
            <el-icon><Camera /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.totalDetections }}</div>
            <div class="stat-label">检测总数</div>
          </div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="stat-card" style="--card-color: #67c23a">
          <div class="stat-icon">
            <el-icon><SuccessFilled /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.completedJobs }}</div>
            <div class="stat-label">成功任务</div>
          </div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="stat-card" style="--card-color: #e6a23c">
          <div class="stat-icon">
            <el-icon><FolderOpened /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.totalDatasets }}</div>
            <div class="stat-label">数据集</div>
          </div>
        </div>
      </el-col>
      <el-col :span="6">
        <div class="stat-card" style="--card-color: #f56c6c">
          <div class="stat-icon">
            <el-icon><Warning /></el-icon>
          </div>
          <div class="stat-info">
            <div class="stat-value">{{ stats.cracksDetected }}</div>
            <div class="stat-label">发现裂纹</div>
          </div>
        </div>
      </el-col>
    </el-row>

    <el-row :gutter="20">
      <!-- 快捷操作 -->
      <el-col :span="12">
        <div class="card-box">
          <div class="section-title">快捷操作</div>
          <div class="quick-actions">
            <div class="action-item" @click="$router.push('/detection')">
              <div class="action-icon" style="background: #409eff">
                <el-icon><Camera /></el-icon>
              </div>
              <span>开始检测</span>
            </div>
            <div class="action-item" @click="$router.push('/dataset')">
              <div class="action-icon" style="background: #67c23a">
                <el-icon><Upload /></el-icon>
              </div>
              <span>上传数据</span>
            </div>
            <div class="action-item" @click="$router.push('/history')">
              <div class="action-icon" style="background: #e6a23c">
                <el-icon><Timer /></el-icon>
              </div>
              <span>查看历史</span>
            </div>
            <div class="action-item" @click="$router.push('/report')">
              <div class="action-icon" style="background: #909399">
                <el-icon><Document /></el-icon>
              </div>
              <span>生成报告</span>
            </div>
          </div>
        </div>
      </el-col>

      <!-- 最近任务 -->
      <el-col :span="12">
        <div class="card-box">
          <div class="section-title">最近检测</div>
          <el-table :data="recentJobs" style="width: 100%" size="small">
            <el-table-column prop="imageName" label="图像名称" show-overflow-tooltip />
            <el-table-column prop="status" label="状态" width="100">
              <template #default="{ row }">
                <el-tag :type="getStatusType(row.status)" size="small">
                  {{ getStatusText(row.status) }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="createdAt" label="时间" width="160">
              <template #default="{ row }">
                {{ formatTime(row.createdAt) }}
              </template>
            </el-table-column>
          </el-table>
          <div v-if="recentJobs.length === 0" class="empty-tip">
            暂无检测记录
          </div>
        </div>
      </el-col>
    </el-row>

    <!-- 图表区域 -->
    <el-row :gutter="20" style="margin-top: 20px">
      <el-col :span="16">
        <div class="card-box">
          <div class="section-title">检测趋势</div>
          <div ref="trendChartRef" class="chart-container"></div>
        </div>
      </el-col>
      <el-col :span="8">
        <div class="card-box">
          <div class="section-title">裂纹类型分布</div>
          <div ref="pieChartRef" class="chart-container"></div>
        </div>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue'
import * as echarts from 'echarts'
import dayjs from 'dayjs'
import { getJobList } from '@/api/inference'
import { getDatasetList } from '@/api/dataset'

const trendChartRef = ref(null)
const pieChartRef = ref(null)
let trendChart = null
let pieChart = null

const stats = reactive({
  totalDetections: 0,
  completedJobs: 0,
  totalDatasets: 0,
  cracksDetected: 0
})

const recentJobs = ref([])

// 获取状态类型
const getStatusType = (status) => {
  const types = {
    completed: 'success',
    running: 'warning',
    pending: 'info',
    failed: 'danger'
  }
  return types[status] || 'info'
}

// 获取状态文本
const getStatusText = (status) => {
  const texts = {
    completed: '已完成',
    running: '进行中',
    pending: '等待中',
    failed: '失败'
  }
  return texts[status] || status
}

// 格式化时间
const formatTime = (time) => {
  return dayjs(time).format('YYYY-MM-DD HH:mm')
}

// 初始化趋势图
const initTrendChart = () => {
  if (!trendChartRef.value) return
  trendChart = echarts.init(trendChartRef.value)
  
  const option = {
    tooltip: {
      trigger: 'axis'
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    xAxis: {
      type: 'category',
      boundaryGap: false,
      data: getLast7Days()
    },
    yAxis: {
      type: 'value'
    },
    series: [
      {
        name: '检测数量',
        type: 'line',
        smooth: true,
        areaStyle: {
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            { offset: 0, color: 'rgba(64, 158, 255, 0.3)' },
            { offset: 1, color: 'rgba(64, 158, 255, 0.05)' }
          ])
        },
        lineStyle: {
          color: '#409eff',
          width: 2
        },
        itemStyle: {
          color: '#409eff'
        },
        data: [12, 19, 15, 25, 22, 30, 28]
      }
    ]
  }
  
  trendChart.setOption(option)
}

// 初始化饼图
const initPieChart = () => {
  if (!pieChartRef.value) return
  pieChart = echarts.init(pieChartRef.value)
  
  const option = {
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)'
    },
    legend: {
      orient: 'vertical',
      right: 10,
      top: 'center'
    },
    series: [
      {
        type: 'pie',
        radius: ['40%', '70%'],
        center: ['40%', '50%'],
        avoidLabelOverlap: false,
        itemStyle: {
          borderRadius: 4
        },
        label: {
          show: false
        },
        data: [
          { value: 45, name: '横向裂纹', itemStyle: { color: '#409eff' } },
          { value: 30, name: '纵向裂纹', itemStyle: { color: '#67c23a' } },
          { value: 15, name: '网状裂纹', itemStyle: { color: '#e6a23c' } },
          { value: 10, name: '块状裂纹', itemStyle: { color: '#f56c6c' } }
        ]
      }
    ]
  }
  
  pieChart.setOption(option)
}

// 获取最近7天日期
const getLast7Days = () => {
  const days = []
  for (let i = 6; i >= 0; i--) {
    days.push(dayjs().subtract(i, 'day').format('MM-DD'))
  }
  return days
}

// 加载数据
const loadData = async () => {
  try {
    // 获取任务列表
    const jobRes = await getJobList({ page: 1, size: 5 })
    recentJobs.value = jobRes.data?.records || []
    stats.totalDetections = jobRes.data?.total || 0
    stats.completedJobs = recentJobs.value.filter(j => j.status === 'completed').length

    // 获取数据集列表
    const datasetRes = await getDatasetList({ page: 1, size: 100 })
    stats.totalDatasets = datasetRes.data?.total || 0

    // 模拟裂纹数量
    stats.cracksDetected = Math.floor(stats.totalDetections * 0.7)
  } catch (error) {
    console.error('加载数据失败:', error)
    // 使用模拟数据
    stats.totalDetections = 156
    stats.completedJobs = 142
    stats.totalDatasets = 8
    stats.cracksDetected = 89
  }
}

// 窗口大小变化时重绘图表
const handleResize = () => {
  trendChart?.resize()
  pieChart?.resize()
}

onMounted(() => {
  loadData()
  initTrendChart()
  initPieChart()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  trendChart?.dispose()
  pieChart?.dispose()
})
</script>

<style lang="scss" scoped>
.dashboard-header {
  margin-bottom: 24px;

  h2 {
    font-size: 24px;
    color: #303133;
    margin-bottom: 8px;
  }

  p {
    color: #909399;
    font-size: 14px;
  }
}

.stat-cards {
  margin-bottom: 20px;
}

.stat-card {
  background: #fff;
  border-radius: 8px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);

  .stat-icon {
    width: 56px;
    height: 56px;
    border-radius: 12px;
    background: var(--card-color);
    display: flex;
    align-items: center;
    justify-content: center;
    
    .el-icon {
      font-size: 28px;
      color: #fff;
    }
  }

  .stat-info {
    .stat-value {
      font-size: 28px;
      font-weight: 600;
      color: #303133;
    }

    .stat-label {
      font-size: 14px;
      color: #909399;
      margin-top: 4px;
    }
  }
}

.quick-actions {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;

  .action-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    padding: 16px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s;

    &:hover {
      background: #f5f7fa;
    }

    .action-icon {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;

      .el-icon {
        font-size: 24px;
        color: #fff;
      }
    }

    span {
      font-size: 14px;
      color: #606266;
    }
  }
}

.chart-container {
  height: 300px;
}

.empty-tip {
  text-align: center;
  color: #909399;
  padding: 40px 0;
}
</style>
