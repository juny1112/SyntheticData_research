% ======================================================================
%  2RC_results.mat 기반 후처리 (NEW)
%  - 주행부하별 파라미터 추출 (SOC90/70/50/30 × 8부하)
%  - PPT 표에 복붙 가능한 엑셀 생성
%  - SOC별 3×2 subplot:
%       • x축: 주행부하(US06~HW2)
%       • 각 subplot: R0,R1,R2,τ1,τ2,RMSE
%       • 범례: 셀들
%
%  가정:
%   • 각 SIM 파일: SOC 4개 × 주행부하 8종 = 32 seg
%   • seg 순서: [SOC90의 8부하, SOC70의 8부하, SOC50의 8부하, SOC30의 8부하]
%   • 부하 순서: US06, UDDS, HWFET, WLTP, CITY1, CITY2, HW1, HW2
%   • 2RC 피팅 결과는 2RC_results.mat 의 all_para_hats 에 저장
%     (행: seg, 열: [R0 R1 R2 tau1 tau2 RMSE exitflag iter])
% ======================================================================
clc; clear; close all;

%% ── 경로 설정 ────────────────────────────────────────────────────────
% SIM_parsed 폴더 (현재 사용 중인 것과 맞춰주기)
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_SOC_moving_cutoff_5_processed\SIM_parsed';
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_Integrated_6_processed\Test4(order3)\SIM_parsed';
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_fresh_integrated_7_Drivingprocessed\SIM_parsed';

save_root = fileparts(folder_SIM);          % ...\SIM_parsed 상위
save_path = fullfile(save_root,'2RC_fitting');

mat_file = fullfile(save_path,'2RC_results.mat');
if ~exist(mat_file,'file')
    error('2RC_results.mat 를 찾을 수 없습니다: %s', mat_file);
end

%% ── 플롯에서 제외할 셀 이름 설정 (시트/구조체 키 이름 기준) ───────
% 예) exclude_cells = {'Cell2_SIM','Cell4_SIM'};
exclude_cells = {'HNE_fresh_8_1', 'HNE_fresh_22_3'};   % 빼고 싶은 셀 있으면 여기 배열에 추가

%% ── 2RC 결과 로드 ────────────────────────────────────────────────────
S = load(mat_file,'all_para_hats');
all_para_hats = S.all_para_hats;

cellNames = fieldnames(all_para_hats);
if isempty(cellNames)
    error('all_para_hats 에 셀 데이터가 없습니다.');
end

% 주행부하 / SOC 정보
loadNames   = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
socVals     = [90 70 50 30];
socRowNames = {'SOC90','SOC70','SOC50','SOC30'};

nLoads = numel(loadNames);     % 8
nSOC   = numel(socVals);       % 4
nSeg_expected = nLoads * nSOC; % 32

%% ── 엑셀 출력 파일 준비 ─────────────────────────────────────────────
xlsx_out = fullfile(save_path,'DrivingLoad_param_for_PPT.xlsx');
if exist(xlsx_out,'file'), delete(xlsx_out); end

%% ── 셀별 파라미터 텐서 저장용 구조체 (SOC-중심 플롯용) ─────────────
param_all_cells = struct;   % param_all_cells.(cellKey) = [4×8×6]

%% ── 셀별로 반복: PPT용 표 + param_all_cells 채우기 ────────────────
for ci = 1:numel(cellNames)
    cellKey = cellNames{ci};
    P = all_para_hats.(cellKey);          % [nSeg × 8] = [R0 R1 R2 tau1 tau2 RMSE exitflag iter]
    [nSeg, nCol] = size(P);

    fprintf('\n=== [%d/%d] 셀 처리: %s (seg=%d) ===\n', ...
        ci, numel(cellNames), cellKey, nSeg);

    if nCol < 6
        warning('(%s) 파라미터 열이 6개 미만입니다. (size=%dx%d)', cellKey, nSeg, nCol);
        continue;
    end

    if nSeg < nSeg_expected
        warning('(%s) seg 수가 32 미만입니다. (seg=%d) → 있는 것까지만 사용', cellKey, nSeg);
    end

    % 32개까지만 사용 (혹시 더 많으면 앞 32개만)
    nUse = min(nSeg, nSeg_expected);
    P = P(1:nUse, :);

    %% 1) SOC×부하×파라미터 텐서 구성
    % paramSOCLoad(soc, load, paramIdx)
    % paramIdx: 1=R0, 2=R1, 3=R2, 4=tau1, 5=tau2, 6=RMSE
    paramSOCLoad = nan(nSOC, nLoads, 6);

    for idx = 1:nUse
        socIdx  = ceil(idx / nLoads);          % 1..4  (SOC90→SOC30)
        loadIdx = idx - (socIdx-1)*nLoads;     % 1..8  (US06→HW2)
        if socIdx>nSOC || loadIdx>nLoads, continue; end

        paramSOCLoad(socIdx, loadIdx, :) = P(idx,1:6);
    end

    % SOC별/부하별 파라미터를 이후 SOC-중심 플롯을 위해 저장
    param_all_cells.(cellKey) = paramSOCLoad;

    %% 2) PPT용 엑셀 테이블 생성 (행=SOC, 열=부하별 R0~tau2)
    %   - R0,R1,R2는 [Ohm]→[mOhm] 변환
    colNames = {};
    dataMat  = [];

    for l = 1:nLoads
        base = loadNames{l};

        R0 = squeeze(paramSOCLoad(:,l,1)) * 1e3;   % [mOhm]
        R1 = squeeze(paramSOCLoad(:,l,2)) * 1e3;   % [mOhm]
        R2 = squeeze(paramSOCLoad(:,l,3)) * 1e3;   % [mOhm]
        t1 = squeeze(paramSOCLoad(:,l,4));         % [s]
        t2 = squeeze(paramSOCLoad(:,l,5));         % [s]

        block = [R0, R1, R2, t1, t2];              % [4×5]
        dataMat  = [dataMat, block]; %#ok<AGROW>

        colNames = [colNames, { ...
            sprintf('%s_R0',   base), ...
            sprintf('%s_R1',   base), ...
            sprintf('%s_R2',   base), ...
            sprintf('%s_tau1', base), ...
            sprintf('%s_tau2', base)}]; %#ok<AGROW>
    end

    Tcell = array2table(dataMat, ...
        'VariableNames', colNames, ...
        'RowNames',      socRowNames);

    % 시트 이름은 셀 키 그대로 사용 (예: Cell1_SIM → Cell1_SIM)
    writetable(Tcell, xlsx_out, ...
        'Sheet', cellKey, ...
        'WriteRowNames', true);

    fprintf('   → 엑셀 시트 작성 완료 (%s)\n', cellKey);
end

fprintf('\n★★ DrivingLoad_param_for_PPT.xlsx 생성 완료 ★★\n');

%% ── SOC별 3×2 subplot: 범례=셀, x축=주행부하 ────────────────────────
% 플롯에 포함할 셀 목록 (exclude_cells 제거)
cells_for_plot = setdiff(fieldnames(param_all_cells), exclude_cells);

if isempty(cells_for_plot)
    warning('플롯에 포함할 셀이 없습니다. exclude_cells 설정을 확인하세요.');
else
    % 공통 설정
    paramTitles = {'R_0 [m\Omega]','R_1 [m\Omega]','R_2 [m\Omega]', ...
                   '\tau_1 [s]','\tau_2 [s]','RMSE [V]'};
    scaleFactor = [1e3, 1e3, 1e3, 1, 1, 1];  % R0~R2만 mOhm로 스케일
    nParam      = numel(paramTitles);

    x = 1:nLoads;

    for s = 1:nSOC   % SOC90,70,50,30 각각에 대해 1장씩
        socVal = socVals(s);

        fig = figure('Name',sprintf('SOC%d – loadwise params (cells)', socVal), ...
                     'NumberTitle','off','Color','w', ...
                     'Position',[100 100 1500 800]);
        tl = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

        for p = 1:nParam
            ax = nexttile; hold(ax,'on'); grid(ax,'on');

            Y_all = [];  % y-range 계산용

            for ci = 1:numel(cells_for_plot)
                cellKey = cells_for_plot{ci};
                paramSOCLoad = param_all_cells.(cellKey);   % [4×8×6]

                if s > size(paramSOCLoad,1)
                    continue;
                end

                Y = squeeze(paramSOCLoad(s,:,p));  % [1×8]
                if all(isnan(Y))
                    continue;
                end

                Y = Y * scaleFactor(p);
                Y_all = [Y_all, Y]; %#ok<AGROW>

                labelStr = strrep(cellKey,'_','\_');
                plot(ax, x, Y, '-o', 'LineWidth',1.5, 'MarkerSize',6, ...
                     'DisplayName', labelStr);
            end

            xlim(ax,[1 nLoads]);
            xticks(ax,1:nLoads);
            xticklabels(ax,loadNames);
            xtickangle(ax,45);

            xlabel(ax,'Driving load');
            ylabel(ax,paramTitles{p},'Interpreter','tex');
            title(ax,paramTitles{p},'Interpreter','tex');

            if p == 1
                legend(ax,'Location','best','Interpreter','tex');
            end

            % y축 자동 범위 (0부터 또는 최소값 근처)
            if ~isempty(Y_all)
                ymax = max(Y_all,[],'omitnan');
                ymin = min(Y_all,[],'omitnan');
                if ~isfinite(ymin) || ymin >= 0
                    ymin = 0;
                end
                if ~isfinite(ymax)
                    ymax = 1;
                else
                    ymax = ymax * 1.05;
                end
                ylim(ax,[ymin ymax]);
            end
        end

        title(tl, sprintf('SOC %d – 2RC params vs driving load (legend = cells)', socVal), ...
              'Interpreter','none');

        fig_base = sprintf('SOC%d_loadwise_params_by_cell', socVal);
        savefig(fig, fullfile(save_path, [fig_base '.fig']));
        exportgraphics(fig, fullfile(save_path, [fig_base '.png']), 'Resolution', 200);

        fprintf('→ SOC %d 플롯 저장 완료: %s\n', ...
            socVal, fullfile(save_path, [fig_base '.png']));
    end

    fprintf('\n★★ SOC별 3×2 subplot (legend=cells) 생성 완료 ★★\n');
end
