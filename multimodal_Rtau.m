% multimodal_Rtau  (bimodal 버전과 동일 로직·표현 유지)
% ────────────────────────────────────────────────────────────────
%   • RC 쌍 총 n개를 선형 τ_i 에 배치
%   • 모드1, 모드2, 모드3 에 배정할 개수(n1,n2,n3) 지정
%   • 각 모드별 normpdf(μ,σ) 계산 → 합이 지정 면적(R_area) 되도록 스케일
%
% [X, R_i, tau_i] = trimodal_Rtau(n, n_mode12, ...
%                                  tau_peak, sigma, R_area, ...
%                                  tau_rng, R0)
%
%   n          : RC 총개수 (예 120)
%   n_mode12   : [n1 n2]  → 모드3 = n - n1 - n2
%   tau_peak   : [μ1 μ2 μ3] (피크 위치 [s])
%   sigma      : [σ1 σ2 σ3] (표준편차 [s])
%   R_area     : [R1 R2 R3] (피크 면적=저항 [Ω])
%   tau_rng    : [τ_min τ_max] (배치 범위 [s])
%   R0         : 직렬 옴성저항 [Ω] (없으면 0)
%
%   반환 X = [R0; R_i; tau_i]  (열벡터)
%
%─────────────────────────────────────────────────────────────────
function [X, R_i, tau_i] = multimodal_Rtau(n, n_mode12, ...
                                         tau_peak, sigma, R_mode_sum, ...
                                         tau_rng, R0)

    if nargin < 7, R0 = 0; end     % 기본값

    n1 = n_mode12(1); % 각 mode 별 n 개수
    n2 = n_mode12(2);
    n3 = n - n1 - n2;

    % ── τ_i 배치 (log scale) ──────────────────────────────────────────
    tau_i = logspace(log10(tau_rng(1)), log10(tau_rng(2)), n).';

    %── 모드별 PDF(log-domain) 계산 ────────────────────────────
    pdf1 = normpdf( log10(tau_i(1:n1)),           log10(tau_peak(1)), sigma(1) );
    pdf2 = normpdf( log10(tau_i(n1+1:n1+n2)),     log10(tau_peak(2)), sigma(2) );
    pdf3 = normpdf( log10(tau_i(n1+n2+1:end)),    log10(tau_peak(3)), sigma(3) );

    % ── 면적(R_area) 기준 스케일링 → R_i 할당 ─────────────────
    R_i                  = zeros(n,1);
    R_i(1:n1)            = R_mode_sum(1) * pdf1 / sum(pdf1);   % Σ = R_area Ω
    R_i(n1+1:n1+n2)      = R_mode_sum(2) * pdf2 / sum(pdf2);
    R_i(n1+n2+1:end)     = R_mode_sum(3) * pdf3 / sum(pdf3);

    % ── 파라미터 벡터 X 반환 ───────────────────────────────────
    X = [R0 ; R_i ; tau_i];
end
