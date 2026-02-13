package com.crack.dataset;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

/**
 * 数据集服务启动类
 */
@SpringBootApplication
@ComponentScan(basePackages = {"com.crack.dataset", "com.crack.common"})
@MapperScan("com.crack.dataset.mapper")
public class DatasetApplication {

    public static void main(String[] args) {
        SpringApplication.run(DatasetApplication.class, args);
        System.out.println("=========================================");
        System.out.println("          数据集服务启动成功！");
        System.out.println("  Dataset Service Started Successfully!");
        System.out.println("=========================================");
    }
}
