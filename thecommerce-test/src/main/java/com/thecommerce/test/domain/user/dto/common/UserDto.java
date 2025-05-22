package com.thecommerce.test.domain.user.dto.common;

import com.thecommerce.test.domain.user.entity.User;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class UserDto {
    private String loginId;
    private String password;
    private String nickname;
    private String name;
    private String phoneNumber;
    private String email;
    private LocalDateTime joinAt;

    public static UserDto toDto(User user){
        return UserDto.builder()
                .loginId(user.getLoginId())
                .password(user.getPassword())
                .nickname(user.getNickname())
                .name(user.getName())
                .phoneNumber(user.getPhoneNumber())
                .email(user.getEmail())
                .joinAt(user.getJoinAt())
                .build();
    }
}
