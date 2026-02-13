<template>
  <div class="page-container">
    <!-- 数据集信息 -->
    <div class="card-box">
      <el-skeleton :loading="loading" animated>
        <template #default>
          <div class="dataset-header" v-if="dataset">
            <div class="dataset-info">
              <h2>{{ dataset.name }}</h2>
              <p>{{ dataset.description || '暂无描述' }}</p>
              <div class="dataset-meta">
                <el-tag>{{ dataset.totalImages }} 张图像</el-tag>
                <el-tag type="success">训练集: {{ dataset.trainCount }}</el-tag>
                <el-tag type="warning">验证集: {{ dataset.valCount }}</el-tag>
                <el-tag type="info">测试集: {{ dataset.testCount }}</el-tag>
              </div>
            </div>
            <div class="dataset-actions">
              <el-button type="primary" @click="showUploadDialog">
                <el-icon><Upload /></el-icon>上传图像
              </el-button>
              <el-button @click="batchDetect">
                <el-icon><VideoPlay /></el-icon>批量检测
              </el-button>
              <el-button @click="goBack">
                <el-icon><Back /></el-icon>返回
              </el-button>
            </div>
          </div>
        </template>
      </el-skeleton>
    </div>

    <!-- 图像列表 -->
    <div class="card-box">
      <div class="section-title">图像列表</div>
      
      <!-- 筛选 -->
      <div class="filter-bar">
        <el-select v-model="filter.splitType" placeholder="数据划分" clearable @change="loadImages">
          <el-option label="全部" value="" />
          <el-option label="训练集" value="train" />
          <el-option label="验证集" value="val" />
          <el-option label="测试集" value="test" />
        </el-select>
        <el-input
          v-model="filter.keyword"
          placeholder="搜索文件名"
          style="width: 200px"
          clearable
          @keyup.enter="loadImages"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
      </div>

      <!-- 图像网格 -->
      <div class="image-grid" v-loading="loadingImages">
        <div 
          v-for="image in imageList" 
          :key="image.id" 
          class="image-item"
          @click="previewImage(image)"
        >
          <el-image 
            :src="image.fileUrl" 
            fit="cover"
            lazy
          >
            <template #error>
              <div class="image-error">
                <el-icon><Picture /></el-icon>
              </div>
            </template>
          </el-image>
          <div class="image-info">
            <span class="image-name">{{ image.filename }}</span>
            <el-tag size="small" :type="getSplitTypeColor(image.splitType)">
              {{ getSplitTypeName(image.splitType) }}
            </el-tag>
          </div>
          <div class="image-actions">
            <el-button type="primary" size="small" circle :icon="View" @click.stop="previewImage(image)" />
            <el-button type="danger" size="small" circle :icon="Delete" @click.stop="handleDeleteImage(image)" />
          </div>
        </div>
      </div>

      <el-empty v-if="!loadingImages && imageList.length === 0" description="暂无图像" />

      <!-- 分页 -->
      <div class="pagination-wrapper" v-if="imageList.length > 0">
        <el-pagination
          v-model:current-page="imagePagination.page"
          v-model:page-size="imagePagination.size"
          :total="imagePagination.total"
          :page-sizes="[20, 40, 60]"
          layout="total, sizes, prev, pager, next"
          @size-change="loadImages"
          @current-change="loadImages"
        />
      </div>
    </div>

    <!-- 上传对话框 -->
    <el-dialog v-model="uploadDialogVisible" title="上传图像" width="600px">
      <el-form label-width="80px">
        <el-form-item label="数据划分">
          <el-radio-group v-model="uploadSplitType">
            <el-radio value="train">训练集</el-radio>
            <el-radio value="val">验证集</el-radio>
            <el-radio value="test">测试集</el-radio>
          </el-radio-group>
        </el-form-item>
      </el-form>
      <el-upload
        class="upload-area"
        drag
        multiple
        :auto-upload="false"
        :on-change="handleUploadChange"
        :file-list="uploadFiles"
        accept="image/*"
      >
        <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
        <div class="el-upload__text">拖拽或点击上传</div>
      </el-upload>
      <template #footer>
        <el-button @click="uploadDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="uploading" @click="handleUpload">
          上传 ({{ uploadFiles.length }} 张)
        </el-button>
      </template>
    </el-dialog>

    <!-- 图像预览 -->
    <el-image-viewer
      v-if="showViewer"
      :url-list="[currentPreviewUrl]"
      @close="showViewer = false"
    />
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { View, Delete } from '@element-plus/icons-vue'
import { 
  getDatasetDetail, 
  getImageList, 
  batchUploadImages,
  deleteImage 
} from '@/api/dataset'
import { batchDetect as apiBatchDetect } from '@/api/inference'

const route = useRoute()
const router = useRouter()

const datasetId = computed(() => route.params.id)

const loading = ref(true)
const loadingImages = ref(false)
const dataset = ref(null)
const imageList = ref([])

const filter = reactive({
  splitType: '',
  keyword: ''
})

const imagePagination = reactive({
  page: 1,
  size: 20,
  total: 0
})

// 上传相关
const uploadDialogVisible = ref(false)
const uploadFiles = ref([])
const uploadSplitType = ref('train')
const uploading = ref(false)

// 预览相关
const showViewer = ref(false)
const currentPreviewUrl = ref('')

// 获取划分类型颜色
const getSplitTypeColor = (type) => {
  const colors = { train: 'success', val: 'warning', test: 'info' }
  return colors[type] || 'info'
}

// 获取划分类型名称
const getSplitTypeName = (type) => {
  const names = { train: '训练', val: '验证', test: '测试' }
  return names[type] || type
}

// 加载数据集详情
const loadDataset = async () => {
  try {
    loading.value = true
    const res = await getDatasetDetail(datasetId.value)
    dataset.value = res.data
  } catch (error) {
    console.error('加载数据集失败:', error)
    ElMessage.error('加载失败')
  } finally {
    loading.value = false
  }
}

// 加载图像列表
const loadImages = async () => {
  try {
    loadingImages.value = true
    const res = await getImageList(datasetId.value, {
      page: imagePagination.page,
      size: imagePagination.size,
      splitType: filter.splitType,
      keyword: filter.keyword
    })
    imageList.value = res.data?.records || []
    imagePagination.total = res.data?.total || 0
  } catch (error) {
    console.error('加载图像失败:', error)
  } finally {
    loadingImages.value = false
  }
}

// 显示上传对话框
const showUploadDialog = () => {
  uploadFiles.value = []
  uploadDialogVisible.value = true
}

// 上传文件变化
const handleUploadChange = (file, fileList) => {
  uploadFiles.value = fileList
}

// 执行上传
const handleUpload = async () => {
  if (uploadFiles.value.length === 0) {
    ElMessage.warning('请选择图片')
    return
  }

  try {
    uploading.value = true
    const formData = new FormData()
    uploadFiles.value.forEach(file => {
      formData.append('files', file.raw)
    })
    formData.append('splitType', uploadSplitType.value)

    await batchUploadImages(datasetId.value, formData)
    ElMessage.success('上传成功')
    uploadDialogVisible.value = false
    loadDataset()
    loadImages()
  } catch (error) {
    console.error('上传失败:', error)
  } finally {
    uploading.value = false
  }
}

// 删除图像
const handleDeleteImage = async (image) => {
  try {
    await ElMessageBox.confirm('确定要删除这张图像吗？', '提示', {
      type: 'warning'
    })
    await deleteImage(datasetId.value, image.id)
    ElMessage.success('删除成功')
    loadImages()
  } catch (error) {
    if (error !== 'cancel') {
      console.error('删除失败:', error)
    }
  }
}

// 预览图像
const previewImage = (image) => {
  currentPreviewUrl.value = image.fileUrl
  showViewer.value = true
}

// 批量检测
const batchDetect = async () => {
  if (imageList.value.length === 0) {
    ElMessage.warning('暂无图像可检测')
    return
  }

  try {
    await ElMessageBox.confirm(
      `确定要对数据集中的 ${imagePagination.total} 张图像进行检测吗？`,
      '批量检测',
      { type: 'info' }
    )

    const imageUrls = imageList.value.map(img => img.fileUrl)
    await apiBatchDetect({
      imageUrls,
      datasetId: datasetId.value
    })
    
    ElMessage.success('批量检测任务已提交')
    router.push('/history')
  } catch (error) {
    if (error !== 'cancel') {
      console.error('批量检测失败:', error)
    }
  }
}

// 返回
const goBack = () => {
  router.push('/dataset')
}

onMounted(() => {
  loadDataset()
  loadImages()
})
</script>

<style lang="scss" scoped>
.dataset-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;

  .dataset-info {
    h2 {
      margin: 0 0 8px 0;
      color: #303133;
    }

    p {
      color: #909399;
      margin: 0 0 12px 0;
    }

    .dataset-meta {
      display: flex;
      gap: 8px;
    }
  }

  .dataset-actions {
    display: flex;
    gap: 8px;
  }
}

.filter-bar {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
}

.image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
}

.image-item {
  position: relative;
  border-radius: 8px;
  overflow: hidden;
  background: #f5f7fa;
  cursor: pointer;
  transition: transform 0.3s;

  &:hover {
    transform: translateY(-4px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);

    .image-actions {
      opacity: 1;
    }
  }

  .el-image {
    width: 100%;
    height: 150px;
  }

  .image-info {
    padding: 8px 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;

    .image-name {
      font-size: 12px;
      color: #606266;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 120px;
    }
  }

  .image-actions {
    position: absolute;
    top: 8px;
    right: 8px;
    display: flex;
    gap: 4px;
    opacity: 0;
    transition: opacity 0.3s;
  }

  .image-error {
    width: 100%;
    height: 150px;
    display: flex;
    justify-content: center;
    align-items: center;
    background: #f5f7fa;
    
    .el-icon {
      font-size: 48px;
      color: #c0c4cc;
    }
  }
}

.pagination-wrapper {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}

.upload-area {
  margin-top: 16px;
}
</style>
