package com.crack.report;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.scheduling.annotation.EnableAsync;

/**
 * 报告服务启动类
 */
@SpringBootApplication
@ComponentScan(basePackages = {"com.crack.report", "com.crack.common"})
@MapperScan("com.crack.report.mapper")
@EnableAsync
public class ReportApplication {

    public static void main(String[] args) {
        SpringApplication.run(ReportApplication.class, args);
        System.out.println("=========================================");
        System.out.println("          报告服务启动成功！");
        System.out.println("   Report Service Started Successfully!");
        System.out.println("=========================================");
    }
}
