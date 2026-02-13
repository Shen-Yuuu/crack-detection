package com.crack.dataset.controller;

import com.crack.common.result.PageResult;
import com.crack.common.result.Result;
import com.crack.dataset.dto.*;
import com.crack.dataset.service.DatasetService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.List;

/**
 * 数据集控制器
 */
@RestController
@RequestMapping("/api/v1/dataset")
@RequiredArgsConstructor
@Tag(name = "数据集管理", description = "数据集CRUD及图像管理接口")
public class DatasetController {

    private final DatasetService datasetService;

    @PostMapping
    @Operation(summary = "创建数据集", description = "创建新的数据集")
    public Result<DatasetResponse> createDataset(
            @Valid @RequestBody CreateDatasetRequest request,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        DatasetResponse response = datasetService.createDataset(request, userId);
        return Result.success("创建成功", response);
    }

    @GetMapping("/{datasetId}")
    @Operation(summary = "获取数据集", description = "获取数据集详情")
    public Result<DatasetResponse> getDataset(
            @Parameter(description = "数据集ID") @PathVariable Long datasetId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        DatasetResponse response = datasetService.getDataset(datasetId, userId);
        return Result.success(response);
    }

    @GetMapping
    @Operation(summary = "数据集列表", description = "获取数据集列表")
    public Result<PageResult<DatasetResponse>> listDatasets(
            DatasetQueryParam param,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        PageResult<DatasetResponse> result = datasetService.listDatasets(param, userId);
        return Result.success(result);
    }

    @DeleteMapping("/{datasetId}")
    @Operation(summary = "删除数据集", description = "删除数据集及其所有图像")
    public Result<Void> deleteDataset(
            @Parameter(description = "数据集ID") @PathVariable Long datasetId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        datasetService.deleteDataset(datasetId, userId);
        return Result.success("删除成功", null);
    }

    @PostMapping("/{datasetId}/images")
    @Operation(summary = "上传图像", description = "上传单张图像到数据集")
    public Result<ImageResponse> uploadImage(
            @Parameter(description = "数据集ID") @PathVariable Long datasetId,
            @Parameter(description = "图像文件") @RequestPart("image") MultipartFile image,
            @Parameter(description = "掩码文件") @RequestPart(value = "mask", required = false) MultipartFile mask,
            @Parameter(description = "数据集划分") @RequestParam(value = "split", defaultValue = "train") String split,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        ImageResponse response = datasetService.uploadImage(datasetId, image, mask, split, userId);
        return Result.success("上传成功", response);
    }

    @PostMapping("/{datasetId}/images/batch")
    @Operation(summary = "批量上传图像", description = "批量上传图像到数据集")
    public Result<List<ImageResponse>> batchUploadImages(
            @Parameter(description = "数据集ID") @PathVariable Long datasetId,
            @Parameter(description = "图像文件列表") @RequestPart("images") List<MultipartFile> images,
            @Parameter(description = "掩码文件列表") @RequestPart(value = "masks", required = false) List<MultipartFile> masks,
            @Parameter(description = "数据集划分") @RequestParam(value = "split", defaultValue = "train") String split,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        List<ImageResponse> responses = datasetService.batchUploadImages(datasetId, images, masks, split, userId);
        return Result.success("上传成功", responses);
    }

    @GetMapping("/{datasetId}/images")
    @Operation(summary = "图像列表", description = "获取数据集中的图像列表")
    public Result<PageResult<ImageResponse>> listImages(
            @Parameter(description = "数据集ID") @PathVariable Long datasetId,
            @Parameter(description = "数据集划分") @RequestParam(value = "split", required = false) String split,
            @Parameter(description = "页码") @RequestParam(value = "page", defaultValue = "1") Integer page,
            @Parameter(description = "每页条数") @RequestParam(value = "size", defaultValue = "10") Integer size,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        PageResult<ImageResponse> result = datasetService.listImages(datasetId, split, page, size, userId);
        return Result.success(result);
    }

    @DeleteMapping("/images/{imageId}")
    @Operation(summary = "删除图像", description = "删除单张图像")
    public Result<Void> deleteImage(
            @Parameter(description = "图像ID") @PathVariable Long imageId,
            @RequestHeader(value = "X-User-Id", defaultValue = "1") Long userId) {

        datasetService.deleteImage(imageId, userId);
        return Result.success("删除成功", null);
    }
}
