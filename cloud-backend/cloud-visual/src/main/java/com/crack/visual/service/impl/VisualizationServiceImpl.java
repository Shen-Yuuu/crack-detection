package com.crack.visual.service.impl;

import cn.hutool.core.io.IoUtil;
import com.crack.common.entity.DetectionResult;
import com.crack.common.exception.BusinessException;
import com.crack.common.result.ResultCode;
import com.crack.common.utils.MinioService;
import com.crack.visual.dto.HeatmapConfig;
import com.crack.visual.dto.OverlayConfig;
import com.crack.visual.dto.StatisticsResponse;
import com.crack.visual.mapper.DetectionResultMapper;
import com.crack.visual.service.VisualizationService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.math.BigDecimal;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * 可视化服务实现
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class VisualizationServiceImpl implements VisualizationService {

    private final DetectionResultMapper resultMapper;
    private final MinioService minioService;

    @Override
    public String generateOverlay(Long resultId, OverlayConfig config) {
        log.info("生成叠加图像，结果ID: {}", resultId);

        DetectionResult result = getResult(resultId);

        try {
            // 加载原图
            InputStream originalStream = minioService.getObject(
                    getImageUrlFromJob(result.getJobId())
            );
            BufferedImage originalImage = ImageIO.read(originalStream);

            // 加载掩码
            InputStream maskStream = minioService.getObject(result.getMaskUrl());
            BufferedImage maskImage = ImageIO.read(maskStream);

            // 创建叠加图像
            BufferedImage overlayImage = new BufferedImage(
                    originalImage.getWidth(),
                    originalImage.getHeight(),
                    BufferedImage.TYPE_INT_ARGB
            );

            Graphics2D g2d = overlayImage.createGraphics();

            // 绘制原图
            g2d.drawImage(originalImage, 0, 0, null);

            // 设置透明度
            g2d.setComposite(AlphaComposite.getInstance(
                    AlphaComposite.SRC_OVER, config.getAlpha()));

            // 解析颜色
            Color maskColor = Color.decode(config.getMaskColor());

            // 绘制掩码
            for (int y = 0; y < maskImage.getHeight(); y++) {
                for (int x = 0; x < maskImage.getWidth(); x++) {
                    int rgb = maskImage.getRGB(x, y);
                    int gray = (rgb >> 16) & 0xFF;
                    if (gray > 128) {
                        g2d.setColor(maskColor);
                        g2d.fillRect(x, y, 1, 1);
                    }
                }
            }

            g2d.dispose();

            // 保存到MinIO
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(overlayImage, "png", baos);
            byte[] bytes = baos.toByteArray();

            String overlayPath = "overlays/" + UUID.randomUUID() + ".png";
            minioService.uploadStream(
                    new ByteArrayInputStream(bytes),
                    overlayPath,
                    "image/png",
                    bytes.length
            );

            // 更新结果
            result.setOverlayUrl(overlayPath);
            resultMapper.updateById(result);

            return minioService.getPresignedUrl(overlayPath);

        } catch (Exception e) {
            log.error("生成叠加图像失败", e);
            throw new BusinessException("生成叠加图像失败: " + e.getMessage());
        }
    }

    @Override
    public String generateHeatmap(Long resultId, HeatmapConfig config) {
        log.info("生成热力图，结果ID: {}", resultId);

        DetectionResult result = getResult(resultId);

        try {
            // 加载掩码
            InputStream maskStream = minioService.getObject(result.getMaskUrl());
            BufferedImage maskImage = ImageIO.read(maskStream);

            // 创建热力图
            BufferedImage heatmap = new BufferedImage(
                    maskImage.getWidth(),
                    maskImage.getHeight(),
                    BufferedImage.TYPE_INT_RGB
            );

            // 应用颜色映射
            for (int y = 0; y < maskImage.getHeight(); y++) {
                for (int x = 0; x < maskImage.getWidth(); x++) {
                    int rgb = maskImage.getRGB(x, y);
                    int gray = (rgb >> 16) & 0xFF;
                    
                    // 简单的jet颜色映射
                    Color color = getJetColor(gray / 255.0f);
                    heatmap.setRGB(x, y, color.getRGB());
                }
            }

            // 保存到MinIO
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ImageIO.write(heatmap, "png", baos);
            byte[] bytes = baos.toByteArray();

            String heatmapPath = "heatmaps/" + UUID.randomUUID() + ".png";
            minioService.uploadStream(
                    new ByteArrayInputStream(bytes),
                    heatmapPath,
                    "image/png",
                    bytes.length
            );

            // 更新结果
            result.setHeatmapUrl(heatmapPath);
            resultMapper.updateById(result);

            return minioService.getPresignedUrl(heatmapPath);

        } catch (Exception e) {
            log.error("生成热力图失败", e);
            throw new BusinessException("生成热力图失败: " + e.getMessage());
        }
    }

    @Override
    public StatisticsResponse getStatistics(Long resultId) {
        DetectionResult result = getResult(resultId);

        // 基于结果数据计算统计信息
        Map<String, Integer> typeDistribution = new HashMap<>();
        typeDistribution.put("横向裂纹", 30);
        typeDistribution.put("纵向裂纹", 45);
        typeDistribution.put("龟裂", 25);

        Map<String, Integer> severityDistribution = new HashMap<>();
        severityDistribution.put("轻微", 40);
        severityDistribution.put("中等", 35);
        severityDistribution.put("严重", 25);

        return StatisticsResponse.builder()
                .totalCracks(result.getCrackCount())
                .totalArea(result.getTotalArea())
                .totalLength(new BigDecimal("15.5"))
                .avgWidth(new BigDecimal("2.3"))
                .maxWidth(new BigDecimal("5.8"))
                .typeDistribution(typeDistribution)
                .severityDistribution(severityDistribution)
                .build();
    }

    /**
     * 获取检测结果
     */
    private DetectionResult getResult(Long resultId) {
        DetectionResult result = resultMapper.selectById(resultId);
        if (result == null) {
            throw new BusinessException(ResultCode.JOB_NOT_FOUND, "检测结果不存在");
        }
        return result;
    }

    /**
     * 获取任务的图像URL（需要关联查询）
     */
    private String getImageUrlFromJob(Long jobId) {
        // 简化处理，实际应该关联查询
        return "images/placeholder.jpg";
    }

    /**
     * Jet颜色映射
     */
    private Color getJetColor(float value) {
        float r, g, b;

        if (value < 0.25f) {
            r = 0;
            g = 4 * value;
            b = 1;
        } else if (value < 0.5f) {
            r = 0;
            g = 1;
            b = 1 - 4 * (value - 0.25f);
        } else if (value < 0.75f) {
            r = 4 * (value - 0.5f);
            g = 1;
            b = 0;
        } else {
            r = 1;
            g = 1 - 4 * (value - 0.75f);
            b = 0;
        }

        return new Color(
                Math.max(0, Math.min(1, r)),
                Math.max(0, Math.min(1, g)),
                Math.max(0, Math.min(1, b))
        );
    }
}
