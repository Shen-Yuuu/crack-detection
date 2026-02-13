package com.crack.dataset.service.impl;

import com.baomidou.mybatisplus.core.conditions.query.LambdaQueryWrapper;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.crack.common.constant.Constants;
import com.crack.common.entity.Dataset;
import com.crack.common.entity.Image;
import com.crack.common.exception.BusinessException;
import com.crack.common.result.PageResult;
import com.crack.common.result.ResultCode;
import com.crack.common.utils.MinioService;
import com.crack.dataset.dto.*;
import com.crack.dataset.mapper.DatasetMapper;
import com.crack.dataset.mapper.ImageMapper;
import com.crack.dataset.service.DatasetService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import org.springframework.util.StringUtils;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 数据集服务实现
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class DatasetServiceImpl implements DatasetService {

    private final DatasetMapper datasetMapper;
    private final ImageMapper imageMapper;
    private final MinioService minioService;

    @Override
    @Transactional(rollbackFor = Exception.class)
    public DatasetResponse createDataset(CreateDatasetRequest request, Long userId) {
        log.info("创建数据集: {}, 用户ID: {}", request.getName(), userId);

        // 检查名称是否已存在
        Long count = datasetMapper.selectCount(new LambdaQueryWrapper<Dataset>()
                .eq(Dataset::getName, request.getName())
                .eq(Dataset::getUserId, userId));
        if (count > 0) {
            throw new BusinessException("数据集名称已存在");
        }

        Dataset dataset = new Dataset();
        dataset.setName(request.getName());
        dataset.setDescription(request.getDescription());
        dataset.setFormat(request.getFormat());
        dataset.setUserId(userId);
        dataset.setStatus(Constants.DATASET_STATUS_PENDING);
        dataset.setTotalImages(0);
        dataset.setTrainCount(0);
        dataset.setValCount(0);
        dataset.setTestCount(0);

        datasetMapper.insert(dataset);

        log.info("数据集创建成功: {}", dataset.getId());
        return convertToResponse(dataset);
    }

    @Override
    public DatasetResponse getDataset(Long datasetId, Long userId) {
        Dataset dataset = getDatasetByIdAndUser(datasetId, userId);
        return convertToResponse(dataset);
    }

    @Override
    public PageResult<DatasetResponse> listDatasets(DatasetQueryParam param, Long userId) {
        Page<Dataset> page = new Page<>(param.getPage(), param.getSize());

        LambdaQueryWrapper<Dataset> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Dataset::getUserId, userId);

        if (StringUtils.hasText(param.getName())) {
            wrapper.like(Dataset::getName, param.getName());
        }
        if (StringUtils.hasText(param.getStatus())) {
            wrapper.eq(Dataset::getStatus, param.getStatus());
        }

        wrapper.orderByDesc(Dataset::getCreatedAt);

        Page<Dataset> resultPage = datasetMapper.selectPage(page, wrapper);

        List<DatasetResponse> records = resultPage.getRecords().stream()
                .map(this::convertToResponse)
                .collect(Collectors.toList());

        return PageResult.of(
                resultPage.getCurrent(),
                resultPage.getSize(),
                resultPage.getTotal(),
                records
        );
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteDataset(Long datasetId, Long userId) {
        log.info("删除数据集: {}, 用户ID: {}", datasetId, userId);

        Dataset dataset = getDatasetByIdAndUser(datasetId, userId);

        // 删除所有图像
        List<Image> images = imageMapper.selectList(new LambdaQueryWrapper<Image>()
                .eq(Image::getDatasetId, datasetId));

        for (Image image : images) {
            try {
                if (StringUtils.hasText(image.getFilePath())) {
                    minioService.deleteObject(image.getFilePath());
                }
                if (StringUtils.hasText(image.getMaskPath())) {
                    minioService.deleteObject(image.getMaskPath());
                }
            } catch (Exception e) {
                log.warn("删除文件失败: {}", e.getMessage());
            }
        }

        imageMapper.delete(new LambdaQueryWrapper<Image>()
                .eq(Image::getDatasetId, datasetId));

        datasetMapper.deleteById(datasetId);

        log.info("数据集删除成功: {}", datasetId);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public ImageResponse uploadImage(Long datasetId, MultipartFile image, MultipartFile mask,
                                     String split, Long userId) {
        log.info("上传图像到数据集: {}", datasetId);

        Dataset dataset = getDatasetByIdAndUser(datasetId, userId);

        // 上传图像
        String folder = Constants.FOLDER_DATASETS + "/" + datasetId + "/" + Constants.FOLDER_IMAGES;
        String imagePath = minioService.uploadFile(image, folder);

        // 上传掩码（如果有）
        String maskPath = null;
        if (mask != null && !mask.isEmpty()) {
            String maskFolder = Constants.FOLDER_DATASETS + "/" + datasetId + "/" + Constants.FOLDER_MASKS;
            maskPath = minioService.uploadFile(mask, maskFolder);
        }

        // 获取图像尺寸
        int width = 0, height = 0;
        try (InputStream is = image.getInputStream()) {
            BufferedImage bufferedImage = ImageIO.read(is);
            if (bufferedImage != null) {
                width = bufferedImage.getWidth();
                height = bufferedImage.getHeight();
            }
        } catch (Exception e) {
            log.warn("读取图像尺寸失败: {}", e.getMessage());
        }

        // 创建图像记录
        Image img = new Image();
        img.setDatasetId(datasetId);
        img.setFileName(image.getOriginalFilename());
        img.setFilePath(imagePath);
        img.setMaskPath(maskPath);
        img.setWidth(width);
        img.setHeight(height);
        img.setFileSize(image.getSize());
        img.setSplit(split != null ? split : "train");

        imageMapper.insert(img);

        // 更新统计信息
        updateStatistics(datasetId);

        return convertToImageResponse(img);
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public List<ImageResponse> batchUploadImages(Long datasetId, List<MultipartFile> images,
                                                  List<MultipartFile> masks, String split, Long userId) {
        log.info("批量上传图像到数据集: {}, 数量: {}", datasetId, images.size());

        List<ImageResponse> responses = new ArrayList<>();

        for (int i = 0; i < images.size(); i++) {
            MultipartFile image = images.get(i);
            MultipartFile mask = (masks != null && i < masks.size()) ? masks.get(i) : null;

            ImageResponse response = uploadImage(datasetId, image, mask, split, userId);
            responses.add(response);
        }

        return responses;
    }

    @Override
    public PageResult<ImageResponse> listImages(Long datasetId, String split,
                                                Integer page, Integer size, Long userId) {
        // 验证数据集权限
        getDatasetByIdAndUser(datasetId, userId);

        Page<Image> pageObj = new Page<>(page, size);

        LambdaQueryWrapper<Image> wrapper = new LambdaQueryWrapper<>();
        wrapper.eq(Image::getDatasetId, datasetId);

        if (StringUtils.hasText(split)) {
            wrapper.eq(Image::getSplit, split);
        }

        wrapper.orderByDesc(Image::getCreatedAt);

        Page<Image> resultPage = imageMapper.selectPage(pageObj, wrapper);

        List<ImageResponse> records = resultPage.getRecords().stream()
                .map(this::convertToImageResponse)
                .collect(Collectors.toList());

        return PageResult.of(
                resultPage.getCurrent(),
                resultPage.getSize(),
                resultPage.getTotal(),
                records
        );
    }

    @Override
    @Transactional(rollbackFor = Exception.class)
    public void deleteImage(Long imageId, Long userId) {
        log.info("删除图像: {}", imageId);

        Image image = imageMapper.selectById(imageId);
        if (image == null) {
            throw new BusinessException(ResultCode.IMAGE_NOT_FOUND);
        }

        // 验证权限
        getDatasetByIdAndUser(image.getDatasetId(), userId);

        // 删除文件
        try {
            if (StringUtils.hasText(image.getFilePath())) {
                minioService.deleteObject(image.getFilePath());
            }
            if (StringUtils.hasText(image.getMaskPath())) {
                minioService.deleteObject(image.getMaskPath());
            }
        } catch (Exception e) {
            log.warn("删除文件失败: {}", e.getMessage());
        }

        imageMapper.deleteById(imageId);

        // 更新统计信息
        updateStatistics(image.getDatasetId());

        log.info("图像删除成功: {}", imageId);
    }

    @Override
    public void updateStatistics(Long datasetId) {
        // 统计各划分的图像数量
        Long trainCount = imageMapper.selectCount(new LambdaQueryWrapper<Image>()
                .eq(Image::getDatasetId, datasetId)
                .eq(Image::getSplit, "train"));

        Long valCount = imageMapper.selectCount(new LambdaQueryWrapper<Image>()
                .eq(Image::getDatasetId, datasetId)
                .eq(Image::getSplit, "val"));

        Long testCount = imageMapper.selectCount(new LambdaQueryWrapper<Image>()
                .eq(Image::getDatasetId, datasetId)
                .eq(Image::getSplit, "test"));

        Long total = trainCount + valCount + testCount;

        // 更新数据集
        Dataset dataset = new Dataset();
        dataset.setId(datasetId);
        dataset.setTotalImages(total.intValue());
        dataset.setTrainCount(trainCount.intValue());
        dataset.setValCount(valCount.intValue());
        dataset.setTestCount(testCount.intValue());
        dataset.setStatus(total > 0 ? Constants.DATASET_STATUS_COMPLETED : Constants.DATASET_STATUS_PENDING);

        datasetMapper.updateById(dataset);
    }

    /**
     * 根据ID和用户获取数据集
     */
    private Dataset getDatasetByIdAndUser(Long datasetId, Long userId) {
        Dataset dataset = datasetMapper.selectOne(
                new LambdaQueryWrapper<Dataset>()
                        .eq(Dataset::getId, datasetId)
                        .eq(Dataset::getUserId, userId)
        );
        if (dataset == null) {
            throw new BusinessException(ResultCode.DATASET_NOT_FOUND);
        }
        return dataset;
    }

    /**
     * 转换为响应对象
     */
    private DatasetResponse convertToResponse(Dataset dataset) {
        return DatasetResponse.builder()
                .id(dataset.getId())
                .name(dataset.getName())
                .description(dataset.getDescription())
                .format(dataset.getFormat())
                .totalImages(dataset.getTotalImages())
                .trainCount(dataset.getTrainCount())
                .valCount(dataset.getValCount())
                .testCount(dataset.getTestCount())
                .status(dataset.getStatus())
                .createdAt(dataset.getCreatedAt())
                .build();
    }

    /**
     * 转换为图像响应对象
     */
    private ImageResponse convertToImageResponse(Image image) {
        return ImageResponse.builder()
                .id(image.getId())
                .datasetId(image.getDatasetId())
                .fileName(image.getFileName())
                .imageUrl(image.getFilePath() != null ? minioService.getPresignedUrl(image.getFilePath()) : null)
                .maskUrl(image.getMaskPath() != null ? minioService.getPresignedUrl(image.getMaskPath()) : null)
                .width(image.getWidth())
                .height(image.getHeight())
                .fileSize(image.getFileSize())
                .split(image.getSplit())
                .createdAt(image.getCreatedAt())
                .build();
    }
}
