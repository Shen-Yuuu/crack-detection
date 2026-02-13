package com.crack.inference;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.scheduling.annotation.EnableAsync;

/**
 * 推理服务启动类
 */
@SpringBootApplication
@ComponentScan(basePackages = {"com.crack.inference", "com.crack.common"})
@MapperScan("com.crack.inference.mapper")
@EnableFeignClients
@EnableAsync
public class InferenceApplication {

    public static void main(String[] args) {
        SpringApplication.run(InferenceApplication.class, args);
        System.out.println("=========================================");
        System.out.println("          推理服务启动成功！");
        System.out.println("  Inference Service Started Successfully!");
        System.out.println("=========================================");
    }
}
