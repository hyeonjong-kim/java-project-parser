package com.thecommerce.test.domain.user.service;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;
import static org.mockito.MockitoAnnotations.initMocks;

import com.thecommerce.test.domain.user.dto.request.JoinUserRequest;
import com.thecommerce.test.domain.user.dto.request.ModifyUserRequest;
import com.thecommerce.test.domain.user.dto.response.GetUserListResponse;
import com.thecommerce.test.domain.user.entity.User;
import com.thecommerce.test.domain.user.repository.UserRepository;
import com.thecommerce.test.global.exception.BusinessExceptionHandler;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.data.domain.Pageable;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;

import java.util.Arrays;
import java.util.Collections;
import java.util.Optional;

public class UserServiceImplTest {

    @Mock
    private UserRepository userRepository;

    @Mock
    private BCryptPasswordEncoder passwordEncoder;

    @InjectMocks
    private UserServiceImpl userService;

    @BeforeEach
    void setUp() {
        initMocks(this);
    }

    @Test
    void addUser_ThrowsException_IfEmailExists() {
        JoinUserRequest joinRequest = new JoinUserRequest("user1", "password", "nickname", "name", "010-1234-5678", "email@example.com");
        when(userRepository.existsByLoginId(joinRequest.getLoginId())).thenReturn(true);

        assertThrows(BusinessExceptionHandler.class, () -> userService.addUser(joinRequest));
    }

    @Test
    void addUser_Success_IfValidRequest() {
        JoinUserRequest joinRequest = new JoinUserRequest("user1", "password", "nickname", "name", "010-1234-5678", "email@example.com");
        when(userRepository.existsByLoginId(joinRequest.getLoginId())).thenReturn(false);

        doAnswer(invocation -> null).when(userRepository).save(any(User.class));

        assertDoesNotThrow(() -> userService.addUser(joinRequest));

        verify(userRepository, times(1)).save(any(User.class));
    }


    @Test
    void updateUser_ThrowsException_IfUserNotFound() {
        when(userRepository.findByLoginId("user1")).thenReturn(Optional.empty());

        assertThrows(BusinessExceptionHandler.class, () -> userService.updateUser("user1", new ModifyUserRequest()));
    }

    @Test
    void updateUser_ThrowsException_IfPasswordNotMatch() {
        User user = new User();
        user.setPassword("encodedPassword");
        when(userRepository.findByLoginId("user1")).thenReturn(Optional.of(user));
        when(passwordEncoder.matches(anyString(), anyString())).thenReturn(false);

        assertThrows(BusinessExceptionHandler.class, () -> userService.updateUser("user1", new ModifyUserRequest()));
    }

    @Test
    void updateUser_Success_IfValidRequest() {
        User user = new User();
        user.setPassword("encodedPassword");
        ModifyUserRequest modifyRequest = new ModifyUserRequest("oldPassword", "newPassword", "nickname", "name", "010-1234-5678", "email@example.com");

        when(userRepository.findByLoginId("user1")).thenReturn(Optional.of(user));
        when(passwordEncoder.matches("oldPassword", user.getPassword())).thenReturn(true);

        // doNothing() 대신 doAnswer() 사용
        doAnswer(invocation -> null).when(userRepository).save(any(User.class));

        assertDoesNotThrow(() -> userService.updateUser("user1", modifyRequest));

        // 메소드 호출 검증
        verify(userRepository).save(any(User.class));
    }
    @Test
    void selectUsers_ThrowsException_IfNoUsers() {
        when(userRepository.findAll((Pageable) any())).thenReturn(null);

        assertThrows(BusinessExceptionHandler.class, () -> userService.selectUsers(0, 10));
    }

    @Test
    void selectUsers_ReturnsGetUserListResponse_WhenUsersExist() {
        User user = User.builder()
                .loginId("user1")
                .password("encodedPassword")
                .name("User One")
                .nickname("Nickname")
                .phoneNumber("01012345678")
                .email("user1@example.com")
                .build();
        Page<User> mockPage = new PageImpl<>(Arrays.asList(user));
        when(userRepository.findAll(any(Pageable.class))).thenReturn(mockPage);

        GetUserListResponse response = userService.selectUsers(0, 10);

        assertNotNull(response);
        assertEquals(1, response.getTotalSize());
        assertEquals(1, response.getUserList().size());
        assertEquals("user1@example.com", response.getUserList().get(0).getEmail());
    }
}
