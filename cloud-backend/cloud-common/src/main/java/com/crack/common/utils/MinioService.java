package com.crack.common.utils;

import io.minio.*;
import io.minio.http.Method;
import io.minio.messages.Item;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

/**
 * MinIO存储服务
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class MinioService {

    private final MinioClient minioClient;

    @Value("${minio.bucket-name:crack-detection}")
    private String defaultBucketName;

    /**
     * 检查桶是否存在
     */
    public boolean bucketExists(String bucketName) {
        try {
            return minioClient.bucketExists(BucketExistsArgs.builder()
                    .bucket(bucketName)
                    .build());
        } catch (Exception e) {
            log.error("检查桶是否存在失败: {}", e.getMessage());
            return false;
        }
    }

    /**
     * 创建桶
     */
    public void createBucket(String bucketName) {
        try {
            if (!bucketExists(bucketName)) {
                minioClient.makeBucket(MakeBucketArgs.builder()
                        .bucket(bucketName)
                        .build());
                log.info("创建桶成功: {}", bucketName);
            }
        } catch (Exception e) {
            log.error("创建桶失败: {}", e.getMessage());
            throw new RuntimeException("创建存储桶失败", e);
        }
    }

    /**
     * 上传文件
     */
    public String uploadFile(MultipartFile file, String folder) {
        return uploadFile(file, defaultBucketName, folder);
    }

    /**
     * 上传文件到指定桶
     */
    public String uploadFile(MultipartFile file, String bucketName, String folder) {
        try {
            createBucket(bucketName);

            String originalFilename = file.getOriginalFilename();
            String extension = originalFilename != null ? 
                    originalFilename.substring(originalFilename.lastIndexOf(".")) : "";
            String objectName = folder + "/" + UUID.randomUUID() + extension;

            minioClient.putObject(PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .stream(file.getInputStream(), file.getSize(), -1)
                    .contentType(file.getContentType())
                    .build());

            log.info("文件上传成功: {}/{}", bucketName, objectName);
            return objectName;
        } catch (Exception e) {
            log.error("文件上传失败: {}", e.getMessage());
            throw new RuntimeException("文件上传失败", e);
        }
    }

    /**
     * 上传输入流
     */
    public String uploadStream(InputStream stream, String objectName, String contentType, long size) {
        return uploadStream(stream, defaultBucketName, objectName, contentType, size);
    }

    /**
     * 上传输入流到指定桶
     */
    public String uploadStream(InputStream stream, String bucketName, String objectName, 
                               String contentType, long size) {
        try {
            createBucket(bucketName);

            minioClient.putObject(PutObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .stream(stream, size, -1)
                    .contentType(contentType)
                    .build());

            log.info("流上传成功: {}/{}", bucketName, objectName);
            return objectName;
        } catch (Exception e) {
            log.error("流上传失败: {}", e.getMessage());
            throw new RuntimeException("流上传失败", e);
        }
    }

    /**
     * 获取文件
     */
    public InputStream getObject(String objectName) {
        return getObject(defaultBucketName, objectName);
    }

    /**
     * 获取指定桶的文件
     */
    public InputStream getObject(String bucketName, String objectName) {
        try {
            return minioClient.getObject(GetObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .build());
        } catch (Exception e) {
            log.error("获取文件失败: {}", e.getMessage());
            throw new RuntimeException("获取文件失败", e);
        }
    }

    /**
     * 获取文件预签名URL
     */
    public String getPresignedUrl(String objectName) {
        return getPresignedUrl(defaultBucketName, objectName, 7, TimeUnit.DAYS);
    }

    /**
     * 获取文件预签名URL（指定过期时间）
     */
    public String getPresignedUrl(String bucketName, String objectName, int duration, TimeUnit unit) {
        try {
            return minioClient.getPresignedObjectUrl(GetPresignedObjectUrlArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .method(Method.GET)
                    .expiry(duration, unit)
                    .build());
        } catch (Exception e) {
            log.error("获取预签名URL失败: {}", e.getMessage());
            throw new RuntimeException("获取预签名URL失败", e);
        }
    }

    /**
     * 删除文件
     */
    public void deleteObject(String objectName) {
        deleteObject(defaultBucketName, objectName);
    }

    /**
     * 删除指定桶的文件
     */
    public void deleteObject(String bucketName, String objectName) {
        try {
            minioClient.removeObject(RemoveObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .build());
            log.info("文件删除成功: {}/{}", bucketName, objectName);
        } catch (Exception e) {
            log.error("删除文件失败: {}", e.getMessage());
            throw new RuntimeException("删除文件失败", e);
        }
    }

    /**
     * 列出文件夹下的所有文件
     */
    public List<String> listObjects(String prefix) {
        return listObjects(defaultBucketName, prefix);
    }

    /**
     * 列出指定桶中文件夹下的所有文件
     */
    public List<String> listObjects(String bucketName, String prefix) {
        List<String> objects = new ArrayList<>();
        try {
            Iterable<Result<Item>> results = minioClient.listObjects(ListObjectsArgs.builder()
                    .bucket(bucketName)
                    .prefix(prefix)
                    .recursive(true)
                    .build());

            for (Result<Item> result : results) {
                objects.add(result.get().objectName());
            }
        } catch (Exception e) {
            log.error("列出文件失败: {}", e.getMessage());
        }
        return objects;
    }

    /**
     * 检查文件是否存在
     */
    public boolean objectExists(String objectName) {
        return objectExists(defaultBucketName, objectName);
    }

    /**
     * 检查指定桶中文件是否存在
     */
    public boolean objectExists(String bucketName, String objectName) {
        try {
            minioClient.statObject(StatObjectArgs.builder()
                    .bucket(bucketName)
                    .object(objectName)
                    .build());
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
