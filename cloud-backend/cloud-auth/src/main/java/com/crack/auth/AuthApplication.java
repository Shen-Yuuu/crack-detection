package com.crack.auth;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.ComponentScan;

/**
 * 认证服务启动类
 */
@SpringBootApplication
@ComponentScan(basePackages = {"com.crack.auth", "com.crack.common"})
@MapperScan("com.crack.auth.mapper")
public class AuthApplication {

    public static void main(String[] args) {
        SpringApplication.run(AuthApplication.class, args);
        System.out.println("=========================================");
        System.out.println("          认证服务启动成功！");
        System.out.println("   Auth Service Started Successfully!");
        System.out.println("=========================================");
    }
}
