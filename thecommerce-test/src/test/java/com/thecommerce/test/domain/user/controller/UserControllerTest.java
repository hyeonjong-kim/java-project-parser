package com.thecommerce.test.domain.user.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.thecommerce.test.domain.user.dto.request.JoinUserRequest;
import com.thecommerce.test.domain.user.dto.request.ModifyUserRequest;
import com.thecommerce.test.domain.user.repository.UserRepository;
import com.thecommerce.test.domain.user.service.UserService;
import com.thecommerce.test.global.common.response.SuccessResponse;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.data.jpa.mapping.JpaMetamodelMappingContext;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.BDDMockito.given;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(UserController.class)
@MockBean(JpaMetamodelMappingContext.class)
public class UserControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private UserRepository userRepository;  // 예시 리포지토리

    @MockBean
    private UserService userService;

    @Autowired
    private ObjectMapper objectMapper;

    @BeforeEach
    void setUp() {
        // Setup necessary mock behaviors here
    }

    @Test
    void testJoinUser() throws Exception {
        JoinUserRequest joinRequest = new JoinUserRequest("loginId", "password", "nickname", "name", "010-xxxx-xxxx", "email@example.com");
        SuccessResponse<Boolean> response = SuccessResponse.<Boolean>builder()
                .data(true)
                .message("회원가입에 성공하였습니다.")
                .status(201)
                .build();

        mockMvc.perform(post("/api/user/join")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(joinRequest)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.message").value("회원가입에 성공하였습니다."));
    }

    @Test
    void testModifyUser() throws Exception {
        String loginId = "loginId";
        ModifyUserRequest modifyRequest = new ModifyUserRequest("oldPassword", "newPassword", "nickname", "name", "010-xxxx-xxxx", "email@example.com");
        given(userService.updateUser(loginId, modifyRequest)).willReturn(null); // Adjust based on actual return type

        mockMvc.perform(put("/api/user/{loginId}", loginId)
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(objectMapper.writeValueAsString(modifyRequest)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.message").value("회원 정보 수정에 성공하였습니다."));
    }

    @Test
    void testGetUserList() throws Exception {
        Integer page = 1;
        Integer pageSize = 10;
        given(userService.selectUsers(page, pageSize)).willReturn(null); // Adjust based on actual return type

        mockMvc.perform(get("/api/user/list")
                        .param("page", page.toString())
                        .param("pageSize", pageSize.toString()))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.message").value("회원 목록 조회에 성공했습니다."));
    }
}
