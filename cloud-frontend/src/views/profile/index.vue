<template>
  <div class="page-container">
    <div class="profile-layout">
      <!-- 左侧用户信息 -->
      <div class="card-box profile-card">
        <div class="avatar-section">
          <el-upload
            class="avatar-uploader"
            :show-file-list="false"
            :before-upload="beforeAvatarUpload"
            :http-request="uploadAvatar"
          >
            <el-avatar :size="100" :src="userInfo.avatar">
              <el-icon :size="50"><User /></el-icon>
            </el-avatar>
            <div class="upload-mask">
              <el-icon><Camera /></el-icon>
            </div>
          </el-upload>
          <h2>{{ userInfo.username }}</h2>
          <p>{{ userInfo.email }}</p>
          <el-tag :type="userInfo.role === 'admin' ? 'danger' : 'primary'">
            {{ userInfo.role === 'admin' ? '管理员' : '普通用户' }}
          </el-tag>
        </div>
        <el-divider />
        <div class="user-stats">
          <div class="stat-item">
            <span class="stat-value">{{ stats.totalDetections }}</span>
            <span class="stat-label">检测次数</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">{{ stats.totalDatasets }}</span>
            <span class="stat-label">数据集</span>
          </div>
          <div class="stat-item">
            <span class="stat-value">{{ stats.totalReports }}</span>
            <span class="stat-label">报告</span>
          </div>
        </div>
        <el-divider />
        <div class="user-info-list">
          <div class="info-item">
            <el-icon><Clock /></el-icon>
            <span>注册时间: {{ formatTime(userInfo.createdAt) }}</span>
          </div>
          <div class="info-item">
            <el-icon><Location /></el-icon>
            <span>最后登录: {{ formatTime(userInfo.lastLoginAt) }}</span>
          </div>
        </div>
      </div>

      <!-- 右侧设置区 -->
      <div class="settings-area">
        <!-- 基本信息 -->
        <div class="card-box">
          <div class="section-title">基本信息</div>
          <el-form :model="profileForm" label-width="100px">
            <el-form-item label="用户名">
              <el-input v-model="profileForm.username" disabled />
            </el-form-item>
            <el-form-item label="邮箱">
              <el-input v-model="profileForm.email" />
            </el-form-item>
            <el-form-item label="昵称">
              <el-input v-model="profileForm.nickname" placeholder="请输入昵称" />
            </el-form-item>
            <el-form-item label="手机号">
              <el-input v-model="profileForm.phone" placeholder="请输入手机号" />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="saving" @click="saveProfile">
                保存修改
              </el-button>
            </el-form-item>
          </el-form>
        </div>

        <!-- 修改密码 -->
        <div class="card-box">
          <div class="section-title">修改密码</div>
          <el-form 
            ref="passwordFormRef" 
            :model="passwordForm" 
            :rules="passwordRules" 
            label-width="100px"
          >
            <el-form-item label="当前密码" prop="oldPassword">
              <el-input 
                v-model="passwordForm.oldPassword" 
                type="password" 
                show-password
                placeholder="请输入当前密码" 
              />
            </el-form-item>
            <el-form-item label="新密码" prop="newPassword">
              <el-input 
                v-model="passwordForm.newPassword" 
                type="password" 
                show-password
                placeholder="请输入新密码" 
              />
            </el-form-item>
            <el-form-item label="确认密码" prop="confirmPassword">
              <el-input 
                v-model="passwordForm.confirmPassword" 
                type="password" 
                show-password
                placeholder="请再次输入新密码" 
              />
            </el-form-item>
            <el-form-item>
              <el-button type="primary" :loading="changingPassword" @click="changePassword">
                修改密码
              </el-button>
            </el-form-item>
          </el-form>
        </div>

        <!-- 系统设置 -->
        <div class="card-box">
          <div class="section-title">系统设置</div>
          <el-form label-width="100px">
            <el-form-item label="检测阈值">
              <el-slider v-model="settings.threshold" :min="0" :max="100" show-input />
            </el-form-item>
            <el-form-item label="TTA推理">
              <el-switch v-model="settings.enableTTA" />
              <span class="setting-tip">开启后可提高精度，但会增加推理时间</span>
            </el-form-item>
            <el-form-item label="结果通知">
              <el-switch v-model="settings.enableNotification" />
              <span class="setting-tip">检测完成后发送浏览器通知</span>
            </el-form-item>
            <el-form-item>
              <el-button type="primary" @click="saveSettings">保存设置</el-button>
            </el-form-item>
          </el-form>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { useUserStore } from '@/stores/user'
import dayjs from 'dayjs'
import { updateProfile, changePassword as apiChangePassword } from '@/api/auth'

const userStore = useUserStore()

const userInfo = computed(() => userStore.userInfo || {})

const saving = ref(false)
const changingPassword = ref(false)
const passwordFormRef = ref(null)

// 用户统计
const stats = reactive({
  totalDetections: 156,
  totalDatasets: 8,
  totalReports: 12
})

// 基本信息表单
const profileForm = reactive({
  username: '',
  email: '',
  nickname: '',
  phone: ''
})

// 密码表单
const passwordForm = reactive({
  oldPassword: '',
  newPassword: '',
  confirmPassword: ''
})

// 密码验证规则
const validateConfirmPassword = (rule, value, callback) => {
  if (value !== passwordForm.newPassword) {
    callback(new Error('两次输入的密码不一致'))
  } else {
    callback()
  }
}

const passwordRules = {
  oldPassword: [
    { required: true, message: '请输入当前密码', trigger: 'blur' }
  ],
  newPassword: [
    { required: true, message: '请输入新密码', trigger: 'blur' },
    { min: 6, message: '密码长度不能少于6位', trigger: 'blur' }
  ],
  confirmPassword: [
    { required: true, message: '请再次输入新密码', trigger: 'blur' },
    { validator: validateConfirmPassword, trigger: 'blur' }
  ]
}

// 系统设置
const settings = reactive({
  threshold: 50,
  enableTTA: false,
  enableNotification: true
})

// 格式化时间
const formatTime = (time) => {
  if (!time) return '-'
  return dayjs(time).format('YYYY-MM-DD HH:mm')
}

// 初始化表单
const initForm = () => {
  profileForm.username = userInfo.value.username || ''
  profileForm.email = userInfo.value.email || ''
  profileForm.nickname = userInfo.value.nickname || ''
  profileForm.phone = userInfo.value.phone || ''

  // 从localStorage加载设置
  const savedSettings = localStorage.getItem('userSettings')
  if (savedSettings) {
    const parsed = JSON.parse(savedSettings)
    Object.assign(settings, parsed)
  }
}

// 头像上传前验证
const beforeAvatarUpload = (file) => {
  const isImage = file.type.startsWith('image/')
  const isLt2M = file.size / 1024 / 1024 < 2

  if (!isImage) {
    ElMessage.error('只能上传图片文件!')
    return false
  }
  if (!isLt2M) {
    ElMessage.error('图片大小不能超过 2MB!')
    return false
  }
  return true
}

// 上传头像
const uploadAvatar = async ({ file }) => {
  try {
    // TODO: 调用上传头像API
    ElMessage.success('头像上传成功')
  } catch (error) {
    console.error('上传头像失败:', error)
  }
}

// 保存基本信息
const saveProfile = async () => {
  try {
    saving.value = true
    await updateProfile({
      email: profileForm.email,
      nickname: profileForm.nickname,
      phone: profileForm.phone
    })
    ElMessage.success('保存成功')
    userStore.getUserInfo()
  } catch (error) {
    console.error('保存失败:', error)
  } finally {
    saving.value = false
  }
}

// 修改密码
const changePassword = async () => {
  const valid = await passwordFormRef.value?.validate()
  if (!valid) return

  try {
    changingPassword.value = true
    await apiChangePassword({
      oldPassword: passwordForm.oldPassword,
      newPassword: passwordForm.newPassword
    })
    ElMessage.success('密码修改成功')
    passwordForm.oldPassword = ''
    passwordForm.newPassword = ''
    passwordForm.confirmPassword = ''
  } catch (error) {
    console.error('修改密码失败:', error)
  } finally {
    changingPassword.value = false
  }
}

// 保存系统设置
const saveSettings = () => {
  localStorage.setItem('userSettings', JSON.stringify(settings))
  ElMessage.success('设置已保存')
}

onMounted(() => {
  initForm()
})
</script>

<style lang="scss" scoped>
.profile-layout {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 20px;

  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
}

.profile-card {
  text-align: center;

  .avatar-section {
    padding: 20px 0;

    .avatar-uploader {
      position: relative;
      display: inline-block;
      cursor: pointer;

      .upload-mask {
        position: absolute;
        top: 0;
        left: 50%;
        transform: translateX(-50%);
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0;
        transition: opacity 0.3s;

        .el-icon {
          font-size: 24px;
          color: #fff;
        }
      }

      &:hover .upload-mask {
        opacity: 1;
      }
    }

    h2 {
      margin: 16px 0 8px;
      font-size: 20px;
      color: #303133;
    }

    p {
      margin: 0 0 12px;
      color: #909399;
      font-size: 14px;
    }
  }

  .user-stats {
    display: flex;
    justify-content: space-around;
    padding: 10px 0;

    .stat-item {
      text-align: center;

      .stat-value {
        display: block;
        font-size: 24px;
        font-weight: 600;
        color: #409eff;
      }

      .stat-label {
        font-size: 12px;
        color: #909399;
      }
    }
  }

  .user-info-list {
    text-align: left;

    .info-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 0;
      font-size: 14px;
      color: #606266;

      .el-icon {
        color: #909399;
      }
    }
  }
}

.settings-area {
  display: flex;
  flex-direction: column;
  gap: 20px;

  .setting-tip {
    margin-left: 12px;
    font-size: 12px;
    color: #909399;
  }
}
</style>
