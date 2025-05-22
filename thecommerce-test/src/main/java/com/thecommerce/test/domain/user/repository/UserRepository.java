package com.thecommerce.test.domain.user.repository;

import com.thecommerce.test.domain.user.entity.User;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Integer> {
    boolean existsByLoginId(String email);
    Optional<User> findByLoginId(String loginId);
    Page<User> findAll(Pageable pageable);
}
