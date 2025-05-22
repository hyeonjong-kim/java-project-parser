package com.thecommerce.test.domain.user.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import javax.persistence.EntityListeners;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ModifyUserResponse {
    private String loginId;
    private String password;
    private String nickname;
    private String name;
    private String phoneNumber;
    private String email;
}
