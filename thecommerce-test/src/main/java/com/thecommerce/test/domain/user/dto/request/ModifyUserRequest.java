package com.thecommerce.test.domain.user.dto.request;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.NoArgsConstructor;

import javax.validation.constraints.Email;
import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;

@Getter
@AllArgsConstructor
@NoArgsConstructor
@Schema(description = "회원 수정을 위한 요청 객체")
public class ModifyUserRequest {
    @NotBlank(message = "기존 회원의 비밀번호를 입력해 주세요.")
    @Size(min = 8, max = 16, message = "비밀번호는 8글자 이상 16글자 이하여야 합니다.")
    @Schema(description = "기존 비밀번호를 입력해주세요." , example = "thecommerce123!")
    private String beforePassword;

    @NotBlank(message = "변경할 회원의 비밀번호를 입력해 주세요.")
    @Size(min = 8, max = 16, message = "비밀번호는 8글자 이상 16글자 이하여야 합니다.")
    @Schema(description = "변경할 비밀번호를 입력해주세요." , example = "thecommerce123!")
    private String afterPassword;

    @NotBlank(message = "변경 혹은 기존 회원 닉네임을 입력해 주세요.")
    @Size(max = 50, message = "닉네임은 크기는 최대 50이어야 합니다.")
    @Schema(description = "변경 혹은 기존 닉네임을 입력해주세요." , example = "더커머스")
    private String nickname;

    @NotBlank(message = "변경 혹은 기존 회원 이름을 입력해 주세요.")
    @Size(max = 50, message = "이름은 크기는 최대 10이어야 합니다.")
    @Schema(description = "변경 혹은 기존 이름을 입력해주세요." , example = "더커머스")
    private String name;

    @NotBlank(message = "변경 혹은 기존 전화번호를 입력해 주세요.")
    @Size(max = 50, message = "전화번호의 크기는 최대 15이어야 합니다.")
    @Schema(description = "변경 혹은 기존 전화번호를 입력해주세요." , example = "010-xxxx-xxxx")
    private String phoneNumber;

    @NotBlank(message = "변경 혹은 기존 회원의 이메일을 입력해주세요.")
    @Email(message = "이메일 형식에 맞춰 입력해주세요.")
    @Size(max = 100, message = "이메일은 크기는 최대 100이어야 합니다.")
    @Schema(description = "변경 혹은 기존 이메일을 입력해주세요." , example = "thecommerce@gmail.com")
    private String email;
}
