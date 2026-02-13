<template>
  <div class="page-container">
    <div class="card-box">
      <div class="section-title">裂纹检测</div>
      
      <!-- 上传区域 -->
      <div class="upload-section">
        <el-upload
          ref="uploadRef"
          class="upload-area"
          drag
          :auto-upload="false"
          :show-file-list="false"
          :on-change="handleFileChange"
          accept="image/*"
        >
          <div v-if="!previewUrl" class="upload-placeholder">
            <el-icon class="upload-icon"><UploadFilled /></el-icon>
            <div class="upload-text">
              <p>将图片拖拽到此处，或 <em>点击上传</em></p>
              <p class="upload-tip">支持 JPG、PNG 格式，单张图片不超过 50MB</p>
            </div>
          </div>
          <div v-else class="preview-container">
            <img :src="previewUrl" class="preview-image" />
            <div class="preview-overlay">
              <el-button type="primary" :icon="Refresh" circle @click.stop="clearImage" />
            </div>
          </div>
        </el-upload>
      </div>

      <!-- 检测配置 -->
      <div class="config-section" v-if="selectedFile">
        <el-form :model="config" label-width="100px">
          <el-row :gutter="20">
            <el-col :span="8">
              <el-form-item label="检测阈值">
                <el-slider v-model="config.threshold" :min="0.1" :max="0.9" :step="0.1" show-stops />
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="TTA增强">
                <el-switch v-model="config.useTta" />
                <el-tooltip content="测试时增强，提高检测精度但会增加处理时间">
                  <el-icon class="help-icon"><QuestionFilled /></el-icon>
                </el-tooltip>
              </el-form-item>
            </el-col>
            <el-col :span="8">
              <el-form-item label="模型版本">
                <el-select v-model="config.modelVersion" placeholder="选择模型">
                  <el-option label="v1.0 (默认)" value="v1.0" />
                  <el-option label="v1.1 (优化)" value="v1.1" />
                </el-select>
              </el-form-item>
            </el-col>
          </el-row>
        </el-form>
      </div>

      <!-- 操作按钮 -->
      <div class="action-section" v-if="selectedFile">
        <el-button type="primary" size="large" :loading="detecting" @click="startDetection">
          <el-icon><VideoPlay /></el-icon>
          {{ detecting ? '检测中...' : '开始检测' }}
        </el-button>
        <el-button size="large" @click="clearImage">
          <el-icon><Delete /></el-icon>
          清除图片
        </el-button>
      </div>

      <!-- 检测进度 -->
      <div class="progress-section" v-if="detecting">
        <el-progress :percentage="progress" :status="progressStatus" :stroke-width="10" />
        <p class="progress-text">{{ progressText }}</p>
      </div>
    </div>

    <!-- 检测结果 -->
    <div class="card-box" v-if="result">
      <div class="section-title">检测结果</div>
      
      <el-row :gutter="20">
        <!-- 原图 -->
        <el-col :span="8">
          <div class="result-card">
            <div class="result-title">原始图像</div>
            <div class="result-image">
              <el-image :src="previewUrl" fit="contain" :preview-src-list="[previewUrl]" />
            </div>
          </div>
        </el-col>
        
        <!-- 掩码图 -->
        <el-col :span="8">
          <div class="result-card">
            <div class="result-title">裂纹掩码</div>
            <div class="result-image">
              <el-image 
                :src="'data:image/png;base64,' + result.maskBase64" 
                fit="contain"
                :preview-src-list="['data:image/png;base64,' + result.maskBase64]"
              />
            </div>
          </div>
        </el-col>
        
        <!-- 叠加图 -->
        <el-col :span="8">
          <div class="result-card">
            <div class="result-title">检测叠加</div>
            <div class="result-image">
              <el-image 
                :src="'data:image/png;base64,' + result.overlayBase64" 
                fit="contain"
                :preview-src-list="['data:image/png;base64,' + result.overlayBase64]"
              />
            </div>
          </div>
        </el-col>
      </el-row>

      <!-- 统计信息 -->
      <div class="stats-section">
        <el-descriptions :column="4" border>
          <el-descriptions-item label="置信度">
            <el-tag :type="getConfidenceType(result.confidence)">
              {{ (result.confidence * 100).toFixed(1) }}%
            </el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="裂纹数量">
            {{ result.crackCount }} 个
          </el-descriptions-item>
          <el-descriptions-item label="裂纹面积">
            {{ (result.totalArea * 100).toFixed(2) }}%
          </el-descriptions-item>
          <el-descriptions-item label="处理时间">
            {{ result.processingTime?.toFixed(2) }} 秒
          </el-descriptions-item>
        </el-descriptions>
      </div>

      <!-- 操作按钮 -->
      <div class="result-actions">
        <el-button type="primary" @click="saveResult">
          <el-icon><Download /></el-icon>保存结果
        </el-button>
        <el-button @click="generateReport">
          <el-icon><Document /></el-icon>生成报告
        </el-button>
        <el-button @click="newDetection">
          <el-icon><Plus /></el-icon>新检测
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive } from 'vue'
import { ElMessage } from 'element-plus'
import { Refresh } from '@element-plus/icons-vue'
import { detectImage } from '@/api/inference'
import { useRouter } from 'vue-router'

const router = useRouter()
const uploadRef = ref(null)

const selectedFile = ref(null)
const previewUrl = ref('')
const detecting = ref(false)
const progress = ref(0)
const progressStatus = ref('')
const progressText = ref('')
const result = ref(null)

const config = reactive({
  threshold: 0.5,
  useTta: true,
  modelVersion: 'v1.0'
})

// 文件变化处理
const handleFileChange = (file) => {
  if (!file.raw.type.startsWith('image/')) {
    ElMessage.error('请上传图片文件')
    return
  }
  
  if (file.raw.size > 50 * 1024 * 1024) {
    ElMessage.error('图片大小不能超过50MB')
    return
  }
  
  selectedFile.value = file.raw
  previewUrl.value = URL.createObjectURL(file.raw)
  result.value = null
}

// 清除图片
const clearImage = () => {
  selectedFile.value = null
  previewUrl.value = ''
  result.value = null
  progress.value = 0
}

// 开始检测
const startDetection = async () => {
  if (!selectedFile.value) {
    ElMessage.warning('请先上传图片')
    return
  }

  detecting.value = true
  progress.value = 0
  progressText.value = '正在上传图片...'

  try {
    const formData = new FormData()
    formData.append('file', selectedFile.value)
    formData.append('threshold', config.threshold)
    formData.append('useTta', config.useTta)
    formData.append('modelVersion', config.modelVersion)

    progress.value = 30
    progressText.value = '正在进行裂纹检测...'

    const res = await detectImage(formData, (event) => {
      if (event.lengthComputable) {
        const percent = Math.round((event.loaded / event.total) * 30)
        progress.value = percent
      }
    })

    progress.value = 100
    progressStatus.value = 'success'
    progressText.value = '检测完成！'

    result.value = res.data
    ElMessage.success('检测完成')

  } catch (error) {
    console.error('检测失败:', error)
    progressStatus.value = 'exception'
    progressText.value = '检测失败'
    ElMessage.error('检测失败，请重试')
  } finally {
    detecting.value = false
  }
}

// 获取置信度类型
const getConfidenceType = (confidence) => {
  if (confidence >= 0.8) return 'danger'
  if (confidence >= 0.5) return 'warning'
  return 'success'
}

// 保存结果
const saveResult = () => {
  // 下载叠加图
  const link = document.createElement('a')
  link.href = 'data:image/png;base64,' + result.value.overlayBase64
  link.download = 'crack_detection_result.png'
  link.click()
  ElMessage.success('结果已保存')
}

// 生成报告
const generateReport = () => {
  router.push({
    path: '/report',
    query: { resultId: result.value.jobId }
  })
}

// 新检测
const newDetection = () => {
  clearImage()
}
</script>

<style lang="scss" scoped>
.upload-section {
  margin-bottom: 24px;
}

.upload-area {
  width: 100%;

  :deep(.el-upload-dragger) {
    width: 100%;
    height: 300px;
    border: 2px dashed #dcdfe6;
    border-radius: 8px;
    transition: all 0.3s;

    &:hover {
      border-color: #409eff;
    }
  }
}

.upload-placeholder {
  height: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;

  .upload-icon {
    font-size: 64px;
    color: #c0c4cc;
    margin-bottom: 16px;
  }

  .upload-text {
    text-align: center;

    p {
      color: #606266;
      margin: 0;

      em {
        color: #409eff;
        font-style: normal;
      }
    }

    .upload-tip {
      color: #909399;
      font-size: 12px;
      margin-top: 8px;
    }
  }
}

.preview-container {
  position: relative;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;

  .preview-image {
    max-width: 100%;
    max-height: 280px;
    object-fit: contain;
  }

  .preview-overlay {
    position: absolute;
    top: 10px;
    right: 10px;
  }
}

.config-section {
  padding: 20px;
  background: #f5f7fa;
  border-radius: 8px;
  margin-bottom: 24px;

  .help-icon {
    margin-left: 8px;
    color: #909399;
    cursor: pointer;
  }
}

.action-section {
  display: flex;
  justify-content: center;
  gap: 16px;
}

.progress-section {
  margin-top: 24px;
  padding: 20px;
  background: #f5f7fa;
  border-radius: 8px;

  .progress-text {
    text-align: center;
    color: #606266;
    margin-top: 12px;
  }
}

.result-card {
  background: #f5f7fa;
  border-radius: 8px;
  padding: 16px;

  .result-title {
    font-weight: 500;
    color: #303133;
    margin-bottom: 12px;
    text-align: center;
  }

  .result-image {
    height: 240px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #fff;
    border-radius: 4px;

    .el-image {
      max-width: 100%;
      max-height: 100%;
    }
  }
}

.stats-section {
  margin-top: 24px;
}

.result-actions {
  margin-top: 24px;
  display: flex;
  justify-content: center;
  gap: 16px;
}
</style>
