<template>
  <div class="page-container">
    <!-- 顶部操作区 -->
    <div class="card-box">
      <div class="header-actions">
        <div class="section-title">数据集管理</div>
        <el-button type="primary" @click="showCreateDialog">
          <el-icon><Plus /></el-icon>创建数据集
        </el-button>
      </div>
    </div>

    <!-- 数据集列表 -->
    <div class="card-box">
      <el-table :data="datasetList" v-loading="loading" style="width: 100%">
        <el-table-column prop="name" label="数据集名称" min-width="180">
          <template #default="{ row }">
            <div class="dataset-name" @click="goDetail(row.id)">
              <el-icon><FolderOpened /></el-icon>
              <span>{{ row.name }}</span>
            </div>
          </template>
        </el-table-column>
        <el-table-column prop="description" label="描述" min-width="200" show-overflow-tooltip />
        <el-table-column prop="totalImages" label="图像数量" width="120" align="center">
          <template #default="{ row }">
            <el-tag type="info">{{ row.totalImages }} 张</el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="source" label="来源" width="120" />
        <el-table-column prop="status" label="状态" width="100" align="center">
          <template #default="{ row }">
            <el-tag :type="row.status === 'active' ? 'success' : 'info'">
              {{ row.status === 'active' ? '正常' : '归档' }}
            </el-tag>
          </template>
        </el-table-column>
        <el-table-column prop="createdAt" label="创建时间" width="180">
          <template #default="{ row }">
            {{ formatTime(row.createdAt) }}
          </template>
        </el-table-column>
        <el-table-column label="操作" width="200" fixed="right">
          <template #default="{ row }">
            <el-button type="primary" link @click="goDetail(row.id)">
              查看
            </el-button>
            <el-button type="primary" link @click="uploadToDataset(row)">
              上传
            </el-button>
            <el-popconfirm
              title="确定要删除这个数据集吗？"
              @confirm="handleDelete(row.id)"
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
          @size-change="loadDatasets"
          @current-change="loadDatasets"
        />
      </div>
    </div>

    <!-- 创建数据集对话框 -->
    <el-dialog v-model="createDialogVisible" title="创建数据集" width="500px">
      <el-form ref="createFormRef" :model="createForm" :rules="createRules" label-width="100px">
        <el-form-item label="名称" prop="name">
          <el-input v-model="createForm.name" placeholder="请输入数据集名称" />
        </el-form-item>
        <el-form-item label="描述" prop="description">
          <el-input
            v-model="createForm.description"
            type="textarea"
            :rows="3"
            placeholder="请输入描述信息"
          />
        </el-form-item>
        <el-form-item label="来源" prop="source">
          <el-select v-model="createForm.source" placeholder="选择数据来源">
            <el-option label="本地上传" value="upload" />
            <el-option label="无人机采集" value="drone" />
            <el-option label="公开数据集" value="public" />
            <el-option label="其他" value="other" />
          </el-select>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="createDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="creating" @click="handleCreate">创建</el-button>
      </template>
    </el-dialog>

    <!-- 上传对话框 -->
    <el-dialog v-model="uploadDialogVisible" title="上传图像" width="600px">
      <el-upload
        ref="uploadRef"
        class="upload-demo"
        drag
        multiple
        :auto-upload="false"
        :on-change="handleUploadChange"
        :file-list="uploadFiles"
        accept="image/*"
      >
        <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
        <div class="el-upload__text">
          将图片拖拽到此处，或 <em>点击上传</em>
        </div>
        <template #tip>
          <div class="el-upload__tip">
            支持 JPG、PNG 格式，可同时上传多张图片
          </div>
        </template>
      </el-upload>
      <template #footer>
        <el-button @click="uploadDialogVisible = false">取消</el-button>
        <el-button type="primary" :loading="uploading" @click="handleUpload">
          上传 ({{ uploadFiles.length }} 张)
        </el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import dayjs from 'dayjs'
import { 
  getDatasetList, 
  createDataset, 
  deleteDataset, 
  batchUploadImages 
} from '@/api/dataset'

const router = useRouter()

const loading = ref(false)
const datasetList = ref([])
const pagination = reactive({
  page: 1,
  size: 10,
  total: 0
})

// 创建相关
const createDialogVisible = ref(false)
const createFormRef = ref(null)
const creating = ref(false)
const createForm = reactive({
  name: '',
  description: '',
  source: 'upload'
})
const createRules = {
  name: [{ required: true, message: '请输入数据集名称', trigger: 'blur' }]
}

// 上传相关
const uploadDialogVisible = ref(false)
const uploadRef = ref(null)
const uploading = ref(false)
const uploadFiles = ref([])
const currentDataset = ref(null)

// 格式化时间
const formatTime = (time) => {
  return dayjs(time).format('YYYY-MM-DD HH:mm')
}

// 加载数据集列表
const loadDatasets = async () => {
  try {
    loading.value = true
    const res = await getDatasetList({
      page: pagination.page,
      size: pagination.size
    })
    datasetList.value = res.data?.records || []
    pagination.total = res.data?.total || 0
  } catch (error) {
    console.error('加载数据集失败:', error)
  } finally {
    loading.value = false
  }
}

// 显示创建对话框
const showCreateDialog = () => {
  createForm.name = ''
  createForm.description = ''
  createForm.source = 'upload'
  createDialogVisible.value = true
}

// 创建数据集
const handleCreate = async () => {
  const valid = await createFormRef.value?.validate()
  if (!valid) return

  try {
    creating.value = true
    await createDataset(createForm)
    ElMessage.success('创建成功')
    createDialogVisible.value = false
    loadDatasets()
  } catch (error) {
    console.error('创建失败:', error)
  } finally {
    creating.value = false
  }
}

// 删除数据集
const handleDelete = async (id) => {
  try {
    await deleteDataset(id)
    ElMessage.success('删除成功')
    loadDatasets()
  } catch (error) {
    console.error('删除失败:', error)
  }
}

// 跳转详情
const goDetail = (id) => {
  router.push(`/dataset/${id}`)
}

// 上传到数据集
const uploadToDataset = (dataset) => {
  currentDataset.value = dataset
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
    ElMessage.warning('请选择要上传的图片')
    return
  }

  try {
    uploading.value = true
    const formData = new FormData()
    uploadFiles.value.forEach(file => {
      formData.append('files', file.raw)
    })

    await batchUploadImages(currentDataset.value.id, formData)
    ElMessage.success('上传成功')
    uploadDialogVisible.value = false
    loadDatasets()
  } catch (error) {
    console.error('上传失败:', error)
  } finally {
    uploading.value = false
  }
}

onMounted(() => {
  loadDatasets()
})
</script>

<style lang="scss" scoped>
.header-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;

  .section-title {
    margin-bottom: 0;
  }
}

.dataset-name {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color: #409eff;

  &:hover {
    text-decoration: underline;
  }
}

.pagination-wrapper {
  margin-top: 20px;
  display: flex;
  justify-content: flex-end;
}
</style>
