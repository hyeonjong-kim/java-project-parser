package com.thecommerce.test.domain.user.dto.response;

import com.thecommerce.test.domain.user.dto.common.UserDto;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.springframework.data.jpa.domain.support.AuditingEntityListener;

import javax.persistence.EntityListeners;
import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class GetUserListResponse {
    Integer page;
    Integer pageSize;
    Integer totalPage;
    Long totalSize;
    List<UserDto> userList;
}
