<template>
  <div class="page-container">
    <div class="card-box">
      <div class="section-title">检测结果详情</div>
      
      <el-skeleton :loading="loading" animated>
        <template #default>
          <div v-if="result">
            <!-- 图像对比 -->
            <el-row :gutter="20">
              <el-col :span="12">
                <div class="image-card">
                  <div class="image-title">原始图像</div>
                  <el-image 
                    :src="result.originalImageUrl" 
                    fit="contain"
                    :preview-src-list="[result.originalImageUrl]"
                  />
                </div>
              </el-col>
              <el-col :span="12">
                <div class="image-card">
                  <div class="image-title">检测结果</div>
                  <el-image 
                    :src="result.overlayImageUrl" 
                    fit="contain"
                    :preview-src-list="[result.overlayImageUrl]"
                  />
                </div>
              </el-col>
            </el-row>

            <!-- 详细统计 -->
            <div class="stats-section">
              <el-descriptions title="检测统计" :column="3" border>
                <el-descriptions-item label="任务ID">{{ result.jobId }}</el-descriptions-item>
                <el-descriptions-item label="置信度">
                  <el-progress 
                    :percentage="result.confidence * 100" 
                    :color="getConfidenceColor(result.confidence)"
                  />
                </el-descriptions-item>
                <el-descriptions-item label="处理时间">{{ result.processingTime }}s</el-descriptions-item>
                <el-descriptions-item label="裂纹数量">{{ result.crackCount }} 个</el-descriptions-item>
                <el-descriptions-item label="裂纹面积">{{ (result.totalArea * 100).toFixed(2) }}%</el-descriptions-item>
                <el-descriptions-item label="严重程度">
                  <el-tag :type="getSeverityType(result.severityLevel)">
                    {{ getSeverityText(result.severityLevel) }}
                  </el-tag>
                </el-descriptions-item>
              </el-descriptions>
            </div>

            <!-- 操作按钮 -->
            <div class="action-buttons">
              <el-button type="primary" @click="downloadResult">
                <el-icon><Download /></el-icon>下载结果
              </el-button>
              <el-button @click="generateHeatmap">
                <el-icon><PictureFilled /></el-icon>生成热力图
              </el-button>
              <el-button @click="createReport">
                <el-icon><Document /></el-icon>生成报告
              </el-button>
              <el-button @click="goBack">
                <el-icon><Back /></el-icon>返回
              </el-button>
            </div>
          </div>
          <el-empty v-else description="未找到检测结果" />
        </template>
      </el-skeleton>
    </div>
  </div>
</template>

<script setup>
import { useRoute, useRouter } from 'vue-router'
import { getDetectionResult } from '@/api/inference'
import { generateHeatmap as apiGenerateHeatmap } from '@/api/visual'
import { ElMessage } from 'element-plus'

const route = useRoute()
const router = useRouter()

const loading = ref(true)
const result = ref(null)

const jobId = computed(() => route.params.jobId)

// 获取置信度颜色
const getConfidenceColor = (confidence) => {
  if (confidence >= 0.8) return '#f56c6c'
  if (confidence >= 0.5) return '#e6a23c'
  return '#67c23a'
}

// 获取严重程度类型
const getSeverityType = (level) => {
  const types = { high: 'danger', medium: 'warning', low: 'success' }
  return types[level] || 'info'
}

// 获取严重程度文本
const getSeverityText = (level) => {
  const texts = { high: '严重', medium: '中等', low: '轻微' }
  return texts[level] || '未知'
}

// 加载结果
const loadResult = async () => {
  try {
    loading.value = true
    const res = await getDetectionResult(jobId.value)
    result.value = res.data
  } catch (error) {
    console.error('加载结果失败:', error)
    ElMessage.error('加载结果失败')
  } finally {
    loading.value = false
  }
}

// 下载结果
const downloadResult = () => {
  if (result.value?.overlayImageUrl) {
    const link = document.createElement('a')
    link.href = result.value.overlayImageUrl
    link.download = `detection_result_${jobId.value}.png`
    link.click()
  }
}

// 生成热力图
const generateHeatmap = async () => {
  try {
    const res = await apiGenerateHeatmap(result.value.id, {})
    ElMessage.success('热力图生成成功')
    // 刷新结果
    await loadResult()
  } catch (error) {
    console.error('生成热力图失败:', error)
  }
}

// 生成报告
const createReport = () => {
  router.push({
    path: '/report',
    query: { resultId: result.value.id }
  })
}

// 返回
const goBack = () => {
  router.back()
}

onMounted(() => {
  loadResult()
})
</script>

<style lang="scss" scoped>
.image-card {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px;
  text-align: center;

  .image-title {
    font-weight: 500;
    margin-bottom: 12px;
    color: #303133;
  }

  .el-image {
    max-height: 400px;
  }
}

.stats-section {
  margin-top: 24px;
}

.action-buttons {
  margin-top: 24px;
  display: flex;
  justify-content: center;
  gap: 16px;
}
</style>
