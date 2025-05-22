package com.thecommerce.test.global.common.response;

import lombok.Builder;
import lombok.Getter;

/**
 * [공통] API Response 결과의 반환 값을 관리
 */
@Getter
public class SuccessResponse<T> {

    // API 응답 코드 Response
    private int status;

    // API 응답 코드 Message
    private String message;

    // API 응답 결과 Response
    private T data;

    /**
     * ApiResponse 생성자
     *
     * @param status SuccessCode
     * @param message SuccessMessage
     * @param data Response Object
     */
    @Builder
    public SuccessResponse(final int status, final String message, final T data) {
        this.status = status;
        this.message = message;
        this.data = data;
    }

}
