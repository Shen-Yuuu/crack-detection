package com.crack.visual.service;

import com.crack.visual.dto.HeatmapConfig;
import com.crack.visual.dto.OverlayConfig;
import com.crack.visual.dto.StatisticsResponse;

/**
 * 可视化服务接口
 */
public interface VisualizationService {

    /**
     * 生成叠加图像
     *
     * @param resultId 结果ID
     * @param config   叠加配置
     * @return 叠加图像URL
     */
    String generateOverlay(Long resultId, OverlayConfig config);

    /**
     * 生成热力图
     *
     * @param resultId 结果ID
     * @param config   热力图配置
     * @return 热力图URL
     */
    String generateHeatmap(Long resultId, HeatmapConfig config);

    /**
     * 获取统计数据
     *
     * @param resultId 结果ID
     * @return 统计数据
     */
    StatisticsResponse getStatistics(Long resultId);
}
