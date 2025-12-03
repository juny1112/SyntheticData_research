% ======================================================================
%  QC2_user (Qc/2) vs 2-RC ECM 파라미터 Scatter + Errorbar
%   - 각 파라미터별(5개) 개별 figure
%
%  • 입력:
%     1) 2RC_results.mat (all_summary, Tbl_ECM_mean 포함)
%         - 경로: folder_SIM 상위 폴더 아래 '2RC_fitting\2RC_results.mat'
%     2) QC2_user : 각 셀의 Qc/2 [Ah] (열 벡터, 길이 = 셀 개수)
%        - 순서는 Tbl_ECM_mean.RowNames (각 셀) 순서와 동일해야 함
%
%  • 출력:
%     - 선택 SOC에서 파라미터 5개(R0,R1,R2,tau1,tau2)의
%       Mean 값을 y축으로, Min/Max를 error bar로 한
%       Qc/2 vs 파라미터 scatter plot (각각 독립 figure 5개)
% ======================================================================
clc; clear; close all;

%% (0) 2RC_results.mat 위치 지정
%  - 2RC fitting 코드에서 사용했던 folder_SIM 과 동일하게 맞추면 됨
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_agedcell_8_processed\SIM_parsed\셀정렬';

% 2RC fitting 코드와 동일한 규칙:
%   save_root = ...\SIM_parsed
%   save_path = ...\SIM_parsed\2RC_fitting
save_root = fileparts(folder_SIM);          % ...\SIM_parsed 상위
save_path = fullfile(save_root,'2RC_fitting');

% 2RC fitting 코드에서 저장했던 결과 파일
mat_file = fullfile(save_path,'2RC_results.mat');
if ~exist(mat_file,'file')
    error('2RC_results.mat 를 찾을 수 없습니다: %s', mat_file);
end
fprintf('2RC 결과 파일: %s\n', mat_file);

%% (1) 2RC_results.mat 로드
Sres = load(mat_file, 'all_summary', 'Tbl_ECM_mean');
if ~isfield(Sres,'all_summary') || ~isfield(Sres,'Tbl_ECM_mean')
    error('2RC_results.mat 에 all_summary 또는 Tbl_ECM_mean 이 없습니다.');
end
all_summary  = Sres.all_summary;    % struct, 필드: 셀별 summary table
Tbl_ECM_mean = Sres.Tbl_ECM_mean;   % table, RowNames: 셀 이름(makeValidName)

% 셀 개수 및 이름(필드명) 가져오기
cellFields = cellstr(Tbl_ECM_mean.Properties.RowNames);   % = all_summary 필드명과 동일
nCells     = numel(cellFields);

fprintf('로드된 셀 개수: %d 개\n', nCells);

%% (2) QC2_user (Qc/2) 값 정의
%  - 이 부분이 "입력"하는 구간
%  - 순서 = Tbl_ECM_mean.RowNames (각 셀 표시 순서) 와 동일해야 함
QC2_user = [56.14
            55.93
            52.55
            50.52
            52.13
            48.53
            56.34
            55.15
            39.89
            55.63
            55.57
            30.16
            56.86];

QC2_user = QC2_user(:);   % column vector
if numel(QC2_user) ~= nCells
    error('QC2_user 길이(%d)와 셀 개수(%d)가 다릅니다. 순서를 다시 맞춰 주세요.', ...
        numel(QC2_user), nCells);
end

%% (3) SOC 선택 (어느 SOC에서의 파라미터를 볼지)
SOC_list   = [90 70 50 30];
socListStr = {'SOC90','SOC70','SOC50','SOC30'};

try
    [idxSel, tf] = listdlg( ...
        'PromptString','어떤 SOC의 파라미터와 Qc/2를 비교할까요?', ...
        'SelectionMode','single', ...
        'ListString',socListStr, ...
        'InitialValue',2);   % 기본 SOC70

    if ~tf
        warning('선택이 취소되어 SOC70을 사용합니다.');
        idxSel = 2;
    end
catch
    % GUI 안될 때 콘솔로 입력
    fprintf('비교할 SOC 선택: 1=SOC90, 2=SOC70, 3=SOC50, 4=SOC30 [기본=2] : ');
    tmp = input('','s');
    v = str2double(tmp);
    if isnan(v) || v<1 || v>4
        v = 2;
    end
    idxSel = v;
end

socSelVal = SOC_list(idxSel);
fprintf('>> QC2_user vs 파라미터 scatter: 선택 SOC = %s (=%d%%)\n', ...
    socListStr{idxSel}, socSelVal);

%% (4) 선택 SOC에서 각 셀의 Mean/Min/Max 파라미터 추출
pNames = {'R0','R1','R2','tau1','tau2'};  % 5개 파라미터
nP     = numel(pNames);

Y_mean = nan(nCells, nP);
Y_min  = nan(nCells, nP);
Y_max  = nan(nCells, nP);

rowTagMean = sprintf('SOC%d_Mean', socSelVal);
rowTagMin  = sprintf('SOC%d_Min',  socSelVal);
rowTagMax  = sprintf('SOC%d_Max',  socSelVal);

for ci = 1:nCells
    fieldName = cellFields{ci};          % all_summary 의 필드명
    Tcell     = all_summary.(fieldName); % 12×6 table (R* mΩ, tau* s, RMSE V)

    for pi = 1:nP
        p = pNames{pi};
        Y_mean(ci,pi) = valOrNaN(Tcell, rowTagMean, p);
        Y_min(ci,pi)  = valOrNaN(Tcell, rowTagMin , p);
        Y_max(ci,pi)  = valOrNaN(Tcell, rowTagMax , p);
    end
end

% errorbar 용 하한/상한 (Mean 기준 비대칭)
Y_err_low  = Y_mean - Y_min;
Y_err_high = Y_max  - Y_mean;

%% (5) 각 파라미터별 독립 figure에 Scatter + Errorbar 플롯
pLabels_disp = { ...
    'R_0 (m\Omega)', ...
    'R_1 (m\Omega)', ...
    'R_2 (m\Omega)', ...
    '\tau_1 (s)', ...
    '\tau_2 (s)'};

% 제목에 쓸 표기 (underscore 포함해서 예쁘게)
pLabels_title = { ...
    'R_0', ...
    'R_1', ...
    'R_2', ...
    '\tau_1', ...
    '\tau_2'};

for pi = 1:nP
    figure('Name', sprintf('%s vs Q_{C/2} @ SOC%d', pLabels_title{pi}, socSelVal), ...
        'Color','w', 'Position',[300+50*pi 300-30*pi 700 500]);

    errorbar(QC2_user, Y_mean(:,pi), Y_err_low(:,pi), Y_err_high(:,pi), ...
        'o', 'LineWidth',1.2, 'MarkerSize',6);

    grid on;
    xlabel('Q_{C/2} [Ah]', 'Interpreter','tex');   % 수식으로 표기
    ylabel(pLabels_disp{pi}, 'Interpreter','tex');
    title(sprintf('%s vs Q_{C/2} @ SOC%d', pLabels_title{pi}, socSelVal), ...
        'Interpreter','tex');

    % --- y축 0부터 시작하도록 설정 ---
    yMaxPlot = max(Y_max(:,pi),[],'omitnan');
    if ~isnan(yMaxPlot) && yMaxPlot>0
        ylim([0, yMaxPlot*1.1]);   % 10% 여유
    else
        ylim([0, 1]);              % 혹시 전부 NaN이면 기본값
    end
end

%% === 보조 함수 =======================================================
function y = valOrNaN(T, rowName, colName)
    if ismember(rowName, T.Properties.RowNames)
        y = T{rowName, colName};
        if isempty(y)
            y = NaN;
        end
    else
        y = NaN;
    end
end
