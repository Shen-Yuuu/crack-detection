package com.crack.visual;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

/**
 * 可视化服务启动类
 */
@SpringBootApplication
@ComponentScan(basePackages = {"com.crack.visual", "com.crack.common"})
@MapperScan("com.crack.visual.mapper")
public class VisualApplication {

    public static void main(String[] args) {
        SpringApplication.run(VisualApplication.class, args);
        System.out.println("=========================================");
        System.out.println("         可视化服务启动成功！");
        System.out.println("   Visual Service Started Successfully!");
        System.out.println("=========================================");
    }
}
