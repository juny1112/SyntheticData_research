% ======================================================================
%  2RC_results_600s.mat 기반 후처리 (NEW)
%  - 주행부하별 파라미터 추출 (SOC90/70/50/30 × 8부하)
%  - PPT 표에 복붙 가능한 엑셀 생성
%      A) 기존 형식: 행=SOC(90/70/50/30), 열=부하별 R0~tau2
%      B) 추가 형식: [선택 SOC] 에 대해
%           행=R0,R1,R2,tau1,tau2 / 열=US06~HW2  (셀당 1시트)
%  - SOC별 3×2 subplot:
%       • x축: 주행부하(US06~HW2)
%       • 각 subplot: R0,R1,R2,τ1,τ2,RMSE
%       • 범례: 셀들
%
%  가정:
%   • 각 SIM 파일: SOC 4개 × 주행부하 8종 = 32 seg
%   • seg 순서: [SOC90의 8부하, SOC70의 8부하, SOC50의 8부하, SOC30의 8부하]
%   • 부하 순서: US06, UDDS, HWFET, WLTP, CITY1, CITY2, HW1, HW2
%   • 2RC 피팅 결과는 2RC_fitting_600s\2RC_results_600s.mat 의 all_para_hats 에 저장
%     (행: seg, 열: [R0 R1 R2 tau1 tau2 RMSE exitflag iter])
% ======================================================================
clc; clear; close all;

%% ── 경로 설정 ────────────────────────────────────────────────────────
% SIM_parsed 폴더 (600s 피팅을 수행한 것과 동일한 곳)
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_fresh_integrated_7_Drivingprocessed\SIM_parsed';

save_root = fileparts(folder_SIM);          % ...\SIM_parsed 상위
save_path = fullfile(save_root,'2RC_fitting_600s');

mat_file = fullfile(save_path,'2RC_results_600s.mat');
if ~exist(mat_file,'file')
    error('2RC_results_600s.mat 를 찾을 수 없습니다: %s', mat_file);
end

%% ── 플롯에서 제외할 셀 이름 설정 (시트/구조체 키 이름 기준) ───────
exclude_cells = {'HNE_fresh_8_1', 'HNE_fresh_22_3'};   % 필요시 수정

%% ── 사용할 SOC 선택 (토글) ──────────────────────────────────────────
socVals      = [90 70 50 30];
socRowNames  = {'SOC90','SOC70','SOC50','SOC30'};
socListStr   = {'SOC90','SOC70','SOC50','SOC30'};

try
    [idxSel, tf] = listdlg( ...
        'PromptString','파라미터를 사용할 SOC를 선택하세요:', ...
        'SelectionMode','single', ...
        'ListString',socListStr, ...
        'InitialValue',2);   % 기본 SOC70
    if ~tf
        warning('선택이 취소되어 SOC70을 사용합니다.');
        idxSel = 2;
    end
catch
    % GUI 안 될 때 대비해서 콘솔 입력
    fprintf('사용할 SOC를 선택하세요: 1=SOC90, 2=SOC70, 3=SOC50, 4=SOC30 [기본=2] : ');
    tmp = input('','s');
    v = str2double(tmp);
    if isnan(v) || v<1 || v>4
        v = 2;
    end
    idxSel = v;
end
socSelName = socListStr{idxSel};
socSelVal  = socVals(idxSel);
fprintf('>> 선택된 SOC: %s (=%d%%)\n', socSelName, socSelVal);

%% ── 2RC 결과 로드 ────────────────────────────────────────────────────
S = load(mat_file,'all_para_hats');
all_para_hats = S.all_para_hats;

cellNames = fieldnames(all_para_hats);
if isempty(cellNames)
    error('all_para_hats 에 셀 데이터가 없습니다.');
end

% 주행부하 / SOC 정보
loadNames   = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};

nLoads = numel(loadNames);     % 8
nSOC   = numel(socVals);       % 4
nSeg_expected = nLoads * nSOC; % 32

%% ── 엑셀 출력 파일 준비 ─────────────────────────────────────────────
% (A) 기존 형식
xlsx_out = fullfile(save_path,'DrivingLoad_param_for_PPT.xlsx');
if exist(xlsx_out,'file'), delete(xlsx_out); end

% (B) 선택 SOC용 셀별 5×8 테이블 (그림 같은 형식)
xlsx_out_SOC = fullfile(save_path, ...
    sprintf('DrivingLoad_param_for_PPT_SOC%d_matrix.xlsx', socSelVal));
if exist(xlsx_out_SOC,'file'), delete(xlsx_out_SOC); end

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

    %% 2-A) (기존) PPT용 엑셀 테이블 생성 (행=SOC, 열=부하별 R0~tau2)
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
            sprintf('%s_\tau1', base), ...
            sprintf('%s_\tau2', base)}]; %#ok<AGROW>
    end

    Tcell = array2table(dataMat, ...
        'VariableNames', colNames, ...
        'RowNames',      socRowNames);

    % 시트 이름은 셀 키 그대로 사용
    writetable(Tcell, xlsx_out, ...
        'Sheet', cellKey, ...
        'WriteRowNames', true);

    %% 2-B) (추가) 선택 SOC에서의 부하별 파라미터 5×8 테이블
    %      행: R0,R1,R2,tau1,tau2 / 열: US06~HW2
    R0_sel = squeeze(paramSOCLoad(idxSel,:,1)) * 1e3;  % [mOhm]
    R1_sel = squeeze(paramSOCLoad(idxSel,:,2)) * 1e3;
    R2_sel = squeeze(paramSOCLoad(idxSel,:,3)) * 1e3;
    t1_sel = squeeze(paramSOCLoad(idxSel,:,4));       % [s]
    t2_sel = squeeze(paramSOCLoad(idxSel,:,5));       % [s]

    dataMat_SOC = [
        R0_sel;
        R1_sel;
        R2_sel;
        t1_sel;
        t2_sel];

    rowNames_SOC = {'R0 [mOhm]','R1 [mOhm]','R2 [mOhm]','tau1 [s]','tau2 [s]'};

    Tcell_SOC = array2table(dataMat_SOC, ...
        'VariableNames', loadNames, ...
        'RowNames',      rowNames_SOC);

    % 셀별 시트 (예: 신품셀 #21 형식)
    writetable(Tcell_SOC, xlsx_out_SOC, ...
        'Sheet', cellKey, ...
        'WriteRowNames', true);

    fprintf('   → 엑셀 시트 작성 완료 (기존+선택 SOC 매트릭스) : %s\n', cellKey);
end

fprintf('\n★★ DrivingLoad_param_for_PPT.xlsx 및 %s 생성 완료 ★★\n', ...
    sprintf('DrivingLoad_param_for_PPT_SOC%d_matrix.xlsx', socSelVal));

%% ── SOC별 3×2 subplot: 범례=셀, x축=주행부하 ────────────────────────
% 플롯에 포함할 셀 목록 (exclude_cells 제거)
cells_for_plot = setdiff(fieldnames(param_all_cells), exclude_cells);

if isempty(cells_for_plot)
    warning('플롯에 포함할 셀이 없습니다. exclude_cells 설정을 확인하세요.');
else
    % 공통 설정
    % 인덱스: 1=R0, 2=R1, 3=R2, 4=tau1, 5=tau2, 6=RMSE
    paramTitles = {'R_0 [m\Omega]','R_1 [m\Omega]','R_2 [m\Omega]', ...
                   '\tau_1 [s]','\tau_2 [s]','RMSE [V]'};
    scaleFactor = [1e3, 1e3, 1e3, 1, 1, 1];  % R0~R2만 mOhm로 스케일
    nParam      = numel(paramTitles);

    % 타일 배치 순서: (행,열)
    %  1  2  -> R0, RMSE
    %  3  4  -> R1, tau1
    %  5  6  -> R2, tau2
    paramOrder = [1 6 2 4 3 5];  % 위 배치를 위한 파라미터 인덱스 순서

    x = 1:nLoads;

    for s = 1:nSOC   % SOC90,70,50,30 각각에 대해 1장씩
        socVal = socVals(s);

        fig = figure('Name',sprintf('SOC%d – loadwise params (cells)', socVal), ...
                     'NumberTitle','off','Color','w', ...
                     'Position',[100 100 1500 800]);
        tl = tiledlayout(3,2,'TileSpacing','compact','Padding','compact');

        for k = 1:nParam
            p  = paramOrder(k);          % 실제 파라미터 인덱스
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

            if p == 1   % 첫 번째(R0)만 범례 표시
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

        fig_base = sprintf('SOC%d_loadwise_params_by_cell_600s', socVal);
        savefig(fig, fullfile(save_path, [fig_base '.fig']));
        exportgraphics(fig, fullfile(save_path, [fig_base '.png']), 'Resolution', 200);

        fprintf('→ SOC %d 플롯 저장 완료: %s\n', ...
            socVal, fullfile(save_path, [fig_base '.png']));
    end

    fprintf('\n★★ SOC별 3×2 subplot (legend=cells, 600s) 생성 완료 ★★\n');
end

