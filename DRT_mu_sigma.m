function [mu10, tau_mean, tau_std, tau_median, mu_n, sigma_n, T_modes] = ...
    DRT_mu_sigma(tau_mode, sigma10)
% DRT_params_from_peak
%   입력:
%     tau_mode : (K×1) 각 모드의 τ 최대값(=로그정규의 mode) [s]
%     sigma10  : (K×1) θ=log10(τ)의 표준편차(데케이드, >0)
%   출력:
%     mu10      : (K×1) θ 평균 = E[log10 τ]
%     tau_mean  : (K×1) τ 평균 = E[τ]
%     tau_std   : (K×1) τ 표준편차 = Std[τ]
%     tau_median: (K×1) τ 중앙값(=기하평균) = 10^mu10  (참고용)
%     mu_n      : (K×1) 자연로그 파라미터 μ_n = E[ln τ]
%     sigma_n   : (K×1) 자연로그 파라미터 σ_n = Std[ln τ]
%     T_modes : 파라미터(행) × 모드(열) 테이블
%
%   사용된 공식 (θ=log10 τ, ln10 = log(10)):
%     1) τ_mode = 10^( mu10 - (ln10)*sigma10^2 )
%        → mu10 = log10(τ_mode) + (ln10)*sigma10^2
%     2) τ 평균:  E[τ] = 10^mu10 * exp( 0.5*(ln10)^2 * sigma10^2 )
%     3) τ 표준편차: Std[τ] = E[τ] * sqrt( exp( (ln10)^2 * sigma10^2 ) - 1 )
%
%   (동일식, 자연로그 파라미터로 쓰면: μ_n = ln10*mu10, σ_n = ln10*sigma10,
%    E[τ] = exp( μ_n + 0.5 σ_n^2 ),  Var[τ] = (e^{σ_n^2}-1) e^{2μ_n+σ_n^2})

    % --- 입력 정리 및 검증 ---
    tau_mode = tau_mode(:);
    sigma10  = sigma10(:);
    K = numel(tau_mode);
    assert(numel(tau_mode)==numel(sigma10), 'tau_mode와 sigma10의 길이가 다릅니다.');
    assert(all(tau_mode>0), 'tau_mode는 양수여야 합니다.');
    assert(all(sigma10>0),  'sigma10은 양수여야 합니다.');

    ln10 = log(10);

    % 1) θ의 평균
    mu10 = log10(tau_mode) + ln10*(sigma10.^2);

    % 자연로그 파라미터(참고/재사용)
    mu_n    = ln10 * mu10;
    sigma_n = ln10 * sigma10;

    % 2) τ 중앙값, 평균, 표준편차
    tau_median = 10.^mu10;
    tau_mean = 10.^mu10 .* exp( 0.5*(ln10^2) * (sigma10.^2) );
    % 동치 표현: tau_mean = 10.^( mu10 + 0.5*ln10*(sigma10.^2) );
    tau_std  = tau_mean .* sqrt( exp( (ln10^2)*(sigma10.^2) ) - 1 );

    % 3) 파라미터(행) × 모드(열) 테이블 생성
    RowNames = {'tau_mode','sigma10','mu10','tau_median','tau_mean','tau_std','mu_n','sigma_n'};
    ModeNames = strcat("Mode_", string(1:K));  % 열 이름: Mode_1, Mode_2, ...

    M = [ tau_mode(:).';         % 1×K
          sigma10(:).';
          mu10(:).';
          tau_median(:).';
          tau_mean(:).';
          tau_std(:).';
          mu_n(:).';
          sigma_n(:).'  ];       % 8×K 숫자 행렬

    % 테이블(행=파라미터, 열=모드)
    T_modes = array2table(M, 'RowNames', RowNames, 'VariableNames', cellstr(ModeNames));
end
