package com.crack.visual.controller;

import com.crack.common.result.Result;
import com.crack.visual.dto.HeatmapConfig;
import com.crack.visual.dto.OverlayConfig;
import com.crack.visual.dto.StatisticsResponse;
import com.crack.visual.service.VisualizationService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

/**
 * 可视化控制器
 */
@RestController
@RequestMapping("/api/v1/visual")
@RequiredArgsConstructor
@Tag(name = "可视化管理", description = "图像可视化相关接口")
public class VisualController {

    private final VisualizationService visualizationService;

    @PostMapping("/overlay/{resultId}")
    @Operation(summary = "生成叠加图", description = "生成裂纹叠加可视化图像")
    public Result<String> generateOverlay(
            @Parameter(description = "结果ID") @PathVariable Long resultId,
            @RequestBody(required = false) OverlayConfig config) {

        if (config == null) {
            config = new OverlayConfig();
        }
        String url = visualizationService.generateOverlay(resultId, config);
        return Result.success("生成成功", url);
    }

    @PostMapping("/heatmap/{resultId}")
    @Operation(summary = "生成热力图", description = "生成裂纹热力图")
    public Result<String> generateHeatmap(
            @Parameter(description = "结果ID") @PathVariable Long resultId,
            @RequestBody(required = false) HeatmapConfig config) {

        if (config == null) {
            config = new HeatmapConfig();
        }
        String url = visualizationService.generateHeatmap(resultId, config);
        return Result.success("生成成功", url);
    }

    @GetMapping("/statistics/{resultId}")
    @Operation(summary = "获取统计数据", description = "获取检测结果的统计数据")
    public Result<StatisticsResponse> getStatistics(
            @Parameter(description = "结果ID") @PathVariable Long resultId) {

        StatisticsResponse response = visualizationService.getStatistics(resultId);
        return Result.success(response);
    }
}
