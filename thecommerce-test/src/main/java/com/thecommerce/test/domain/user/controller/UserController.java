package com.thecommerce.test.domain.user.controller;

import com.thecommerce.test.domain.user.dto.request.JoinUserRequest;
import com.thecommerce.test.domain.user.dto.request.ModifyUserRequest;
import com.thecommerce.test.domain.user.dto.response.GetUserListResponse;
import com.thecommerce.test.domain.user.dto.response.ModifyUserResponse;
import com.thecommerce.test.domain.user.service.UserService;
import com.thecommerce.test.global.common.code.SuccessCode;
import com.thecommerce.test.global.common.response.ErrorResponse;
import com.thecommerce.test.global.common.response.SuccessResponse;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

@Validated
@RequiredArgsConstructor
@RestController
@RequestMapping("/api/user")
@Tag(name = "회원 API", description = "회원 관리 용 API")
public class UserController {

    private final UserService userService;

    @PostMapping("/join")
    @Operation(summary = "회원가입", description = "회원 정보를 입력하여 회원가입을 합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "201", description = "회원가입에 성공했습니다."),
            @ApiResponse(responseCode = "400", description = "잘못된 요청입니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
            @ApiResponse(responseCode = "409", description = "이미 존재하는 아이디입니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
            @ApiResponse(responseCode = "500", description = "서버 내부 문제입니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    })
    public ResponseEntity<SuccessResponse<?>> join(@Validated @RequestBody JoinUserRequest request) {
        userService.addUser(request);
        SuccessResponse<?> response = SuccessResponse.builder()
                .data(true)
                .status(SuccessCode.INSERT_SUCCESS.getStatus())
                .message("회원가입에 성공하였습니다.")
                .build();
        return new ResponseEntity<>(response, HttpStatus.OK);
    }

    @PutMapping("/{loginId}")
    @Operation(summary = "회원 정보 수정", description = "회원 정보를 수정합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "204", description = "회원 정보 수정에 성공했습니다."),
            @ApiResponse(responseCode = "400", description = "잘못된 요청입니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
            @ApiResponse(responseCode = "404", description = "존재하지 않는 회원입니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
            @ApiResponse(responseCode = "410", description = "비밀번호가 일치하지 않습니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
            @ApiResponse(responseCode = "500", description = "서버 내부 문제입니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    })
    public ResponseEntity<SuccessResponse<?>> modifyUser(@PathVariable String loginId, @Validated @RequestBody ModifyUserRequest request) {
        ModifyUserResponse modifyUserResponse = userService.updateUser(loginId, request);
        SuccessResponse<?> response = SuccessResponse.builder()
                .status(SuccessCode.UPDATE_SUCCESS.getStatus())
                .message("회원 정보 수정에 성공하였습니다.")
                .data(modifyUserResponse)
                .build();
        return new ResponseEntity<>(response, HttpStatus.OK);
    }

    @GetMapping("/list")
    @Operation(summary = "회원 목록 조회", description = "회원 목록 조회를 수행합니다.")
    @ApiResponses(value = {
            @ApiResponse(responseCode = "200", description = "회원 목록 조회에 성공했습니다."),
            @ApiResponse(responseCode = "400", description = "잘못된 요청입니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
            @ApiResponse(responseCode = "404", description = "회원이 존재하지 않습니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class))),
            @ApiResponse(responseCode = "500", description = "서버 내부 문제입니다.", content = @Content(schema = @Schema(implementation = ErrorResponse.class)))
    })
    public ResponseEntity<SuccessResponse<?>> getUserList(@RequestParam Integer page, @RequestParam Integer pageSize) {
        GetUserListResponse getUserListResponse = userService.selectUsers(page, pageSize);
        SuccessResponse<?> response = SuccessResponse.builder()
                .status(SuccessCode.SELECT_SUCCESS.getStatus())
                .message("회원 목록 조회에 성공했습니다.")
                .data(getUserListResponse)
                .build();
        return new ResponseEntity<>(response, HttpStatus.OK);
    }
}
