package com.crack.dataset.service;

import com.crack.common.result.PageResult;
import com.crack.dataset.dto.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * 数据集服务接口
 */
public interface DatasetService {

    /**
     * 创建数据集
     */
    DatasetResponse createDataset(CreateDatasetRequest request, Long userId);

    /**
     * 获取数据集详情
     */
    DatasetResponse getDataset(Long datasetId, Long userId);

    /**
     * 获取数据集列表
     */
    PageResult<DatasetResponse> listDatasets(DatasetQueryParam param, Long userId);

    /**
     * 删除数据集
     */
    void deleteDataset(Long datasetId, Long userId);

    /**
     * 上传图像到数据集
     */
    ImageResponse uploadImage(Long datasetId, MultipartFile image, MultipartFile mask, 
                              String split, Long userId);

    /**
     * 批量上传图像
     */
    List<ImageResponse> batchUploadImages(Long datasetId, List<MultipartFile> images, 
                                          List<MultipartFile> masks, String split, Long userId);

    /**
     * 获取数据集中的图像列表
     */
    PageResult<ImageResponse> listImages(Long datasetId, String split, 
                                         Integer page, Integer size, Long userId);

    /**
     * 删除图像
     */
    void deleteImage(Long imageId, Long userId);

    /**
     * 更新数据集统计信息
     */
    void updateStatistics(Long datasetId);
}
