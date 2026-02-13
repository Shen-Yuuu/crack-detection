package com.crack.inference.client;

import com.crack.inference.dto.PythonInferenceResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * Python推理服务客户端
 */
@FeignClient(name = "python-inference", url = "${python.inference.url}")
public interface PythonInferenceClient {

    /**
     * 单张图像检测
     */
    @PostMapping(value = "/api/v1/inference/detect", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    PythonInferenceResponse detect(
            @RequestPart("file") MultipartFile file,
            @RequestParam(value = "threshold", defaultValue = "0.5") Float threshold,
            @RequestParam(value = "use_tta", defaultValue = "false") Boolean useTta,
            @RequestParam(value = "return_mask", defaultValue = "true") Boolean returnMask,
            @RequestParam(value = "return_overlay", defaultValue = "true") Boolean returnOverlay
    );

    /**
     * 健康检查
     */
    @GetMapping("/health")
    Object healthCheck();
}
