% bimodal_Rtau function
%  ────────────────────────────────────────────────────────────────────
%  각 모드별 R_i 합 = 1 mΩ 로 맞춘 n-RC 파라미터 (τ 선형 스케일, normpdf 사용)
%  ────────────────────────────────────────────────────────────────────
function X = bimodal_Rtau(n_RC, n_mode1, tau_peak, sigma, tau_rng, R0)
    % n_RC     : RC 쌍 총개수 (예 20)
    % n_mode1  : 모드1에 할당할 RC 개수 (예 10)  → 모드2 = n_RC - n_mode1
    % tau_peak : [μ1 μ2]  (피크 위치, [s])
    % sigma    : [σ1 σ2]  (표준편차,  [s])
    % tau_rng  : [τ_min τ_max] (τ 범위, [s])
    % R0       : 직렬 옴성저항 (Ω)        (없으면 0)
    %
    % 반환 X   : [R0, R1…Rn, τ1…τn]  (열 벡터)
    %
    % -- 예시 호출:
    %   X = bimodal_RC_params_exact(20,10,[6 60],[1 10],[0 80],0);

    % ── 기본 파라미터 ───────────────────────────────────────────────
    if nargin < 6,  R0 = 0;  end
    area_mode = 1e-3;                     % 목표: 모드당 1 mΩ

    % ── τ_i 배치 (선형 간격) ─────────────────────────────────────────
    tau_i = linspace(tau_rng(1), tau_rng(2), n_RC).';   % 열 벡터

    % ── 가중치 계산 -------------------------------------------------
    %   • 모드1 : 첫 n_mode1 개 → μ1, σ1 사용
    %   • 모드2 : 나머지        → μ2, σ2 사용
    pdf1 = normpdf(tau_i(1:n_mode1),   tau_peak(1), sigma(1));
    pdf2 = normpdf(tau_i(n_mode1+1:end), tau_peak(2), sigma(2));

    % ── 저항 R_i : 가중치 비율 × 1 mΩ --------------------------------
    R_i               = zeros(n_RC,1);
    R_i(1:n_mode1)    = area_mode * pdf1 / sum(pdf1);     % Σ = 1 mΩ
    R_i(n_mode1+1:end)= area_mode * pdf2 / sum(pdf2);     % Σ = 1 mΩ

    % ── 파라미터 벡터 X 반환 ----------------------------------------
    X = [R0 ; R_i ; tau_i];
end
