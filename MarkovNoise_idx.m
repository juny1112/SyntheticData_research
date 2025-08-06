%% Markov noise function

function [noisy, states, final_state, P, epsilon_vector, eps_values] = MarkovNoise_idx(original, epsilon_percent_span, initial_state, sigma)
%--------------------------------------------------------------------------
% MarkovNoise
%   노이즈 넣을 값 original(전류, 전압 등)에 대해 마르코프 체인을 이용해 상대적 퍼센트 잡음을 추가한 
%   noisy를 생성합니다.
%
%   사용 예시:
%       [noisy, states, final_state, P, epsilon_vector, eps_values] = MarkovNoise(V, 4, 50, 0.001);
%
%   입력값 (Parameters)
%       original             : 원본 벡터 (1차원)
%       epsilon_percent_span : 잡음 편차 범위 (%) 예: 2 ⇒ 약 ±2% 범위
%       initial_state        : 마르코프 체인 시작 상태 (1 ~ N 사이 정수)
%                            (미리 원하는 인덱스로 지정 가능)
%       sigma                : 상태 간 전이 폭(표준편차) 설정 (임의로 설정, 작으면 다른 상태로 전이할 확률 작아짐)
%
%   출력값 (Returns)
%       noisy      : 마르코프 잡음이 추가된 벡터
%       states     : 각 시간 스텝에서의 마르코프 상태 (1 ~ N)
%       final_state: 전체 시뮬레이션 종료 후 마지막 상태
%       P          : 전이 확률 행렬 (N x N)
%       epsilon_vector : 정의된 상태별 잡음 비율 벡터
%       epsilon_values : 각 시간 스텝에서 선택된 잡음 비율 값 배열
%
%   내부 구성
%       - N개의 상태를 균등 간격으로 정의(epsilon_vector)
%       - 각 상태에서 다른 상태로 전이할 때, 상태별 분포(정규분포)에 따른
%         확률밀도 함수를 활용하여 전이 확률 행렬 P를 구성
%       - 매 시간 스텝마다 현재 상태를 바탕으로 전류에 잡음 추가,
%         이후 전이 확률을 통해 다음 상태로 이동
%
%   주의사항
%       - normpdf 함수를 사용하므로, MATLAB Statistics & Machine Learning 
%         Toolbox가 필요합니다(기본 normpdf를 지원하지 않을 경우, 
%         exp(-(x-mu)^2/(2*sigma^2)) / (sigma*sqrt(2*pi)) 형태로 직접 대체 가능).
%         ** normpdf(x, mu, sigma); x=확률계산할 값(상태),mu=평균(현재상태),sigma=표준편차;   
%            => 현재상태를 중심으로한 정규분포 확률을 얻음
%
%   작성자: (필요 시 작성)
%   작성일: (필요 시 작성)
%--------------------------------------------------------------------------

    %----------------------------------------------------------------------
    % 1) 상태(State) 정의
    %   - N개의 상태를 -epsilon_percent_span/2 ~ epsilon_percent_span/2 사이에서 균등 분할
    %   - 예: epsilon_percent_span = 2 이면 약 ±2% (= ±0.02)범위
    %----------------------------------------------------------------------
    N = 101;  % 상태 개수
    epsilon_vector = linspace(-epsilon_percent_span/(100), epsilon_percent_span/(100), N);
    
    %----------------------------------------------------------------------
    % 2) 전이 확률 행렬 P 구성
    %   - 상태 i에서 j로 갈 때의 확률 = 정규분포 pdf(확률밀도함수) 기반
    %   - 각 행(P(i,:))의 합이 1이 되도록 정규화 (합한게 1이 안될수도 있으니까 sum(probabilities)로 나눔. )
    %   - epsilon_vector(i): 현재상태. 이걸 정규분포에서 평균이라고 잡음
    %----------------------------------------------------------------------
    P = zeros(N);
    idx = 1:N;
    for i = 1:N
        probabilities = normpdf(idx, i, sigma);
        P(i, :) = probabilities / sum(probabilities);
    end

    %----------------------------------------------------------------------
    % 3) 마르코프 체인 시뮬레이션
    %   - 초기 상태는 initial_state
    %   - 각 샘플마다 원본 original(k)에 비례한 잡음 추가
    %   - 다음 상태로 전이
    %   ** randsample(population, k, replacement, weights); 
    %   population=선택할수 있는 값, k=뽑을샘플개수, replacement=true면 중복 허용(여기서는 하나만 뽑아서 중복 여부 중요X), 
    %   weights=각 값들이 선택될 확률
    %----------------------------------------------------------------------
    current_state = initial_state;
    noisy = zeros(size(original));
    states  = zeros(size(original));
    eps_values = zeros(size(original)); % 각 스텝의 eps_k 저장

    for k = 1:length(original)
        % 현재 상태 인덱스에 해당하는 잡음 비율
        eps_k = epsilon_vector(current_state);
        eps_values(k) = eps_k;  % 적용된 노이즈 저장

        % 원본 데이터에 잡음 비율만큼 더함
        %   ex) noisy(k) = original(k) + ( |original(k)| * eps_k )
        noisy(k) = original(k) + abs(original(k)) * eps_k;

        % 현재 상태 기록
        states(k) = current_state;

        % 다음 상태로 랜덤 전이
        current_state = randsample(1:N, 1, true, P(current_state, :));
    end

    %----------------------------------------------------------------------
    % 4) 마지막 상태 기록
    %----------------------------------------------------------------------
    final_state = states(end);

end
