function [theta, r_mode, r_theta, dth, R, tau, g_tau, A_modes, w_used] = DRT_Rtau( ...
        n, tau_min, tau_max, mu10, sigma10, A_tot, w)

%   θ=log10(τ) 축에서 다중 모드 가우시안 혼합으로 r(θ)을 직접 생성.
%   필요 시 g(τ)=r/(τ ln10)로 변환.
%
% 입력:
%   n        : 격자 점수 (권장 400~1000)
%   tau_min  : τ 최소값 (>0)
%   tau_max  : τ 최대값 (>tau_min)
%   mu10     : (K×1) 각 모드 평균(데케이드, θ=log10 τ)
%   sigma10  : (K×1) 각 모드 표준편차(데케이드, >0)
%   A_tot    : 총 면적(Ω) → ∫ r(θ) dθ = A_tot
%   w        : (K×1) 면적 비율(>=0). 합이 1이 아니어도 내부 정규화.
%
% 출력:
%   theta    : (n×1) θ=log10(τ)
%   r_mode   : (n×K) 각 모드 r_k(θ) [Ω/decade]
%   r_theta  : (n×1) r(θ)=∑ r_k [Ω/decade]
%   dth      : (n×1) Δθ (등간격)
%   R        : (n×1) r·Δθ (ΣR ≈ A_tot) [Ω]
%   tau      : (n×1) τ = 10.^θ
%   g_tau    : (n×1) g(τ)=r/(τ ln10) [Ω/s]
%   A_modes  : (K×1) 내부에서 계산된 각 모드 면적(Ω)
%   w_used   : (K×1) 정규화 후 사용된 비율


    % --- 비율 정규화 및 모드 면적 ---
    w_used  = w / sum(w);
    A_modes = A_tot * w_used;

    % 1) θ 격자 만들기 (모든 모드 포함하도록 범위 설정)
    th_min = log10(tau_min);
    th_max = log10(tau_max);
    theta  = linspace(th_min, th_max, n).';

    % 2) Δθ: 등간격이므로 상수
    dth0 = (th_max - th_min)/(n - 1);
    dth  = dth0 * ones(n,1);

    % 3) θ: normal distribution (θ ~ N(μ_θ,σ_θ))
    K = numel(mu10);
    r_mode = zeros(n, K);
    for k = 1:K
        % pdf(θ) = (1/(σ_k√(2π))) * exp(-(θ-μ_k)^2/(2σ_k^2))
        % pdf_theta = (1/(sigma10(k)*sqrt(2*pi))) * exp( - (theta-mu10(k)).^2 / (2*sigma10(k)^2) );
        pdf_theta = normpdf(theta, mu10(k), sigma10(k));
        
        % 4) 재정규화: ∑ pdf·Δθ = 1  (등간격이므로 Δθ = dth0 사용)
        pdf_theta = pdf_theta / sum(pdf_theta * dth0);
 
        % 5) 면적(Ω) 적용 → r_k(θ)
        r_mode(:,k) = A_modes(k) * pdf_theta;
    end

    % 6) 합성 r(θ), 저항 R, τ·g 변환
    r_theta   = sum(r_mode, 2);           % Ω/decade (행끼리 합산)
    R         = r_theta .* dth;                  % Ω
    tau       = 10.^theta;               % s
    g_tau     = r_theta ./ (tau * log(10));    % Ω/s

    % (선택) 면적 보존 체크:
    % sumR = sum(R);  % ≈ A_tot
    % sum_g  = sum(g_tau .* dtau); % ≈ A_tot
end
