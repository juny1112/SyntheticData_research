% ======================================================================
%  (개정) 전체 SIM 기반 2-RC 피팅 + SOC(90·70·50·30) 전역 통계
%  • 입력: *_SIM.mat (SIM_table 필요, 총 32 seg = 8×4 가정)
%  • 그룹핑: 기본=앞에서부터 8개씩 [90,70,50,30], 보조=SOC_center 최근접
%  • 출력: 
%       - 파일별 요약 all_summary (12×6: 각 SOC의 Mean/Min/Max)
%           * R0,R1,R2 : mΩ
%           * tau1,tau2 : s
%           * RMSE : V
%       - ECM 평균 파라미터 테이블 Tbl_ECM_mean (R*는 mΩ)
%       - SOC×(avg/min/max) vs 셀 이름 RMSE 테이블 Tbl_RMSE (mV)
%       - (NEW) 선택 SOC에서 R0~τ2 avg/min/max를 셀별로 정리한 엑셀
% ======================================================================
clc; clear; close all;

% ── 경로 & 파일 리스트 ───────────────────────────────────────────────
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_SOC_moving_cutoff_5_processed\SIM_parsed';
% folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_Integrated_6_processed\Test4(order3)\SIM_parsed';
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\0degC\이름정렬';

sim_files  = dir(fullfile(folder_SIM,"*_SIM.mat"));
if isempty(sim_files)
    error("SIM 파일을 찾지 못했습니다: %s", folder_SIM);
end

% 저장 경로는 상위(SIM_parsed) 폴더로 고정
save_root = fileparts(folder_SIM);          % -> ...\SIM_parsed
save_path = fullfile(save_root,'2RC_fitting');
if ~exist(save_path,'dir'); mkdir(save_path); end

% ── fmincon + MultiStart 설정 ────────────────────────────────────────
ms       = MultiStart("UseParallel",true,"Display","off");
startPts = RandomStartPointSet('NumStartPoints',20);
opt      = optimset('display','off','MaxIter',1e3,'MaxFunEvals',1e4, ...
                    'TolFun',1e-14,'TolX',1e-15);

% 2-RC 초기추정값 / 경계 / 선형제약 (τ1<τ2)
para0 = [0.003 0.0005 0.0005 10 100]; 
lb    = [0       0       0      0.01  0.01];
ub    = [0.05 0.005 0.03 100 5000];
A_lin = [0 0 0 1 -1];  b_lin = 0;

% ── 누적 컨테이너 ────────────────────────────────────────────────────
all_para_hats = struct;   % 각 파일: [nSeg × 8] = [R0 R1 R2 tau1 tau2 | RMSE exitflag iter] (R*는 Ω)
all_rmse      = struct;   % 각 파일: [nSeg × 1] RMSE (V)
all_summary   = struct;   % 각 파일: 12×6 요약 테이블 (R0 R1 R2 tau1 tau2 RMSE; 각 SOC의 Mean/Min/Max)
                          %           R*는 mΩ, tau*는 s, RMSE는 V

% 대표 SOC(정리/플롯 기준)
soc_targets  = [50 70];
soc_labels   = ["SOC90","SOC70","SOC50","SOC30"];
rowNames     = { ...
  'SOC90_Mean','SOC90_Min','SOC90_Max', ...
  'SOC70_Mean','SOC70_Min','SOC70_Max', ...
  'SOC50_Mean','SOC50_Min','SOC50_Max', ...
  'SOC30_Mean','SOC30_Min','SOC30_Max'};

% ── 메인 루프 (모든 파일 처리) ───────────────────────────────────────
for f = 1:numel(sim_files)
    % 1) load SIM_table
    S = load(fullfile(folder_SIM,sim_files(f).name),"SIM_table");
    if ~isfield(S,"SIM_table")
        warning("SIM_table 없음: %s (건너뜀)", sim_files(f).name);
        continue
    end
    SIM_table = S.SIM_table;

    base_raw   = erase(sim_files(f).name,"_SIM.mat");
    base_field = matlab.lang.makeValidName(base_raw);

    % 2) 이번 실험: 각 파일에 SIM 총 32개(=8×4) 가정
    nSeg = height(SIM_table);
    if nSeg==0
        warning("No SIM rows: %s", base_raw);
        continue
    end

    % ── 그룹 코드 할당 ────────────────────────────────────────────────
    % 우선순위 1) 8개씩 블록 매핑: [1..8]→90, [9..16]→70, [17..24]→50, [25..32]→30
    grp_code = zeros(nSeg,1);    % 1:90, 2:70, 3:50, 4:30
    if nSeg >= 32
        blk = [1 8; 9 16; 17 24; 25 32];
        for g = 1:4
            ii = blk(g,1):min(blk(g,2), nSeg);
            grp_code(ii) = g;
        end
    end

    % 우선순위 2) 예외(32개가 아니거나 순서 불명확) → SOC_center 최근접 매핑
    if any(grp_code==0)
        SOC_center = mean([SIM_table.SOC1 SIM_table.SOC2], 2, 'omitnan');
        miss = isnan(SOC_center);
        if any(miss) && ismember('SOC_vec', SIM_table.Properties.VariableNames)
            try
                SOC_center(miss) = cellfun(@(v) mean(v,'omitnan'), SIM_table.SOC_vec(miss));
            catch
                % 남은 NaN은 그대로 두고 제외
            end
        end
        valid = ~isnan(SOC_center);
        if any(valid)
            [~, gmin] = min(abs(SOC_center(valid) - soc_targets), [], 2);
            vidx = find(valid);
            grp_code(vidx) = gmin;
        end
        % 여전히 0이면(정보 부족) 해당 세그먼트는 후속 통계에서 자동 제외됨
    end

    % 3) 전 세그먼트 피팅
    para_hats = nan(nSeg, 5+3);   % [R0 R1 R2 tau1 tau2 RMSE exitflag iter]
    RMSE_list = nan(nSeg, 1);     % RMSE (V)

    % 서브플롯 레이아웃(최대 8열 고정, 행은 자동)
    cols = 8;
    rows = max(1, ceil(nSeg/cols));
    fig = figure('Name',[base_raw ' – 2RC fitting'], 'NumberTitle','off', ...
        'Position',[100 100 1600 900], 'Color','w');
    try
        sgtitle(strrep(base_raw,'_','\_') + " – 2RC fitting", 'Interpreter','tex');
    catch
        % (older MATLAB)
        suptitle(strrep(base_raw,'_','\_') + " – 2RC fitting");
    end

    for s = 1:nSeg
        try
            t = SIM_table.time{s};        % duration
            I = SIM_table.current{s};
            V = SIM_table.voltage{s};
            O = []; 
            if ismember('OCV_vec', SIM_table.Properties.VariableNames)
                O = SIM_table.OCV_vec{s};
            end

            problem = createOptimProblem('fmincon', ...
              'objective',@(p)RMSE_2RC(V,p,t,I,O), ...
              'x0',para0,'lb',lb,'ub',ub, ...
              'Aineq',A_lin,'bineq',b_lin,'options',opt);

            [Pbest, Fval, exitflg, ~, sol] = run(ms,problem,startPts);
            it = NaN;
            if ~isempty(sol)
                it = sol(find([sol.Fval]==Fval,1)).Output.iterations;
            end

            para_hats(s,:) = [Pbest, Fval, exitflg, it];  % R*는 Ω, tau*는 s
            RMSE_list(s)   = Fval;                        % V

            % ---- SOC 라벨 생성 ----
            if grp_code(s) >= 1 && grp_code(s) <= numel(soc_targets)
                soc_txt = sprintf('SOC %d', soc_targets(grp_code(s)));
            else
                soc_txt = 'SOC ?';
            end

            % ---- 모델 전압 ----
            V_fit = RC_model_2(Pbest, t, I, O);
            V_ini = RC_model_2(para0 , t, I, O);

            % ---- 서브플롯 ----
            subplot(rows, cols, s);
            plot(t, V, 'k', t, V_fit, 'r', t, V_ini, '--b', 'LineWidth', 1.1);
            grid on;
            xlabel('Time'); ylabel('Voltage (V)');
            ttl = sprintf('Load %d | %s | RMSE=%.2f mV', s, soc_txt, Fval*1e3);
            title(ttl, 'Interpreter','none');
            legend('True','Fitted','Initial','Location','northeast','Box','off');

        catch ME
            warning("(%s) seg %d 피팅 실패: %s", base_raw, s, ME.message);
        end
    end

    % === (A) 파일별 피팅 figure 저장 ===
    if isgraphics(fig,'figure')
        savefig(fig, fullfile(save_path, [base_raw '_2RC_fit.fig']));
    else
        warning('(%s) figure 핸들이 유효하지 않아 저장을 건너뜁니다.', base_raw);
    end
    % 필요하면 창 닫기:
    % close(fig);

    % 4) SOC(90/70/50/30)별 요약 테이블(12×6) 구성
    %    * R0,R1,R2 : mΩ 로 변환
    T = table( ...
        nan(12,1), nan(12,1), nan(12,1), nan(12,1), nan(12,1), nan(12,1), ...
        'VariableNames', {'R0','R1','R2','tau1','tau2','RMSE'}, ...
        'RowNames',     rowNames );

    P_all = para_hats(:,1:5);   % Ω, s

    % 그룹 마스크
    m90 = (grp_code==1);
    m70 = (grp_code==2);
    m50 = (grp_code==3);
    m30 = (grp_code==4);
    groups = {m90,m70,m50,m30};

    r = 1;
    for g = 1:4
        idx = groups{g};
        if any(idx)
            blockP = P_all(idx,:);       % [*,5] = R0 R1 R2 tau1 tau2 (Ω, s)
            blockE = RMSE_list(idx);     % [*,1] = RMSE (V)
            block6 = [blockP, blockE];   % [*,6]

            T{r  ,:} = mean(block6,1,'omitnan');     % Mean
            T{r+1,:} = min (block6,[],1);            % Min
            T{r+2,:} = max (block6,[],1);            % Max
        end
        r = r + 3;
    end

    % ---- 저항 값을 mΩ 로 변환 (R0,R1,R2) ----
    T{:, {'R0','R1','R2'}} = 1000 * T{:, {'R0','R1','R2'}};

    % 5) 누적 저장
    all_para_hats.(base_field) = para_hats;  % R*는 Ω
    all_rmse.(base_field)      = RMSE_list;  % V
    all_summary.(base_field)   = T;          % R*는 mΩ

    % 로그
    fprintf('[done] %s → fitted %d segs, summary(12×6) 저장  |  counts: 90=%d,70=%d,50=%d,30=%d\n', ...
        base_raw, nSeg, nnz(m90), nnz(m70), nnz(m50), nnz(m30));
    
end

fprintf("모든 파일 처리 완료!\n");

%% === ECM 평균 파라미터 테이블 (행: 셀, 열: SOC별 ECM Mean 파라미터) ===
%   - SOC 90 → 70 → 50 → 30 순
%   - R0, R1, R2 는 mΩ, tau1·tau2 는 그대로 (초 단위)

cells_in_results = fieldnames(all_summary);   % fitting 성공한 셀들
keys = strings(0,1);

% SIM 파일 순서대로 정렬된 셀 이름 리스트 생성
for f = 1:numel(sim_files)
    base_raw   = erase(sim_files(f).name,"_SIM.mat");      % 원래 파일명 (확장자 제외)
    base_field = matlab.lang.makeValidName(base_raw);      % struct 필드 이름
    if ismember(base_field, cells_in_results)
        keys(end+1) = string(base_raw);    % 처리된 셀만 순서대로 추가
    end
end

if isempty(keys)
    warning('ECM 테이블 생성 실패: all_summary 가 비어 있습니다.');
else
    cellRawNames = cellstr(keys);      % 사람이 읽는 셀 이름(파일명 기반)
    nCells       = numel(cellRawNames);

    SOC_list = [90 70 50 30];         % 열 그룹 순서
    pNames   = {'R0','R1','R2','tau1','tau2'};
    nP       = numel(pNames);

    ECM_mean_matrix = nan(nCells, numel(SOC_list)*nP);
    varNames        = strings(1, numel(SOC_list)*nP);

    col = 0;
    for si = 1:numel(SOC_list)
        soc = SOC_list(si);
        rowTag = sprintf('SOC%d_Mean', soc);     % 예: 'SOC90_Mean'

        for pi = 1:nP
            col = col + 1;
            p   = pNames{pi};

            % 열 이름: SOC90_R0_mOhm, SOC90_R1_mOhm, ..., SOC70_tau2, ...
            if pi <= 3
                varNames(col) = sprintf('SOC%d_%s_mOhm', soc, p);  % R0,R1,R2 (mΩ)
            else
                varNames(col) = sprintf('SOC%d_%s', soc, p);      % tau1,tau2 (s)
            end

            % 각 셀에 대해 Mean 값 채우기
            for ci = 1:nCells
                key_field = matlab.lang.makeValidName(cellRawNames{ci});
                T = all_summary.(key_field);        % 이 셀의 12×6 요약 table (R*는 mΩ)

                val = valOrNaN(T, rowTag, p);       % 해당 SOC Mean 의 파라미터 (이미 mΩ 또는 s)
                ECM_mean_matrix(ci, col) = val;
            end
        end
    end

    varNames       = cellstr(varNames);
    rowNames_cells = matlab.lang.makeValidName(cellRawNames);  % RowNames 용

    Tbl_ECM_mean = array2table(ECM_mean_matrix, ...
        'RowNames',      rowNames_cells, ...
        'VariableNames', varNames);

    Tbl_ECM_mean.Properties.Description = ...
        '행: 셀(파일명) / 열: SOC별 ECM Mean 파라미터 (R*는 mΩ, tau*는 초)';

    %% === SOC별 RMSE 요약 테이블 (단위: mV) ===========================
    %   - 행: SOC90_avg, SOC90_min, SOC90_max, SOC70_..., SOC50_..., SOC30_...
    %   - 열: 셀 이름(파일명 기반)

    nSOC      = numel(SOC_list);   % 4 (90,70,50,30)
    statNames = ["avg","min","max"];
    nStat     = numel(statNames);  % 3

    RMSE_rowNames = strings(nSOC*nStat,1);
    for si = 1:nSOC
        soc      = SOC_list(si);
        rowBase  = (si-1)*nStat;
        RMSE_rowNames(rowBase+1) = sprintf('SOC%d_avg', soc);
        RMSE_rowNames(rowBase+2) = sprintf('SOC%d_min', soc);
        RMSE_rowNames(rowBase+3) = sprintf('SOC%d_max', soc);
    end

    RMSE_matrix = nan(nSOC*nStat, nCells);

    for ci = 1:nCells
        key_field = matlab.lang.makeValidName(cellRawNames{ci});
        Tcell     = all_summary.(key_field);   % 이 셀의 12×6 요약 table (RMSE는 V 단위)

        for si = 1:nSOC
            soc     = SOC_list(si);
            rowBase = (si-1)*nStat;

            tagMean = sprintf('SOC%d_Mean', soc);
            tagMin  = sprintf('SOC%d_Min',  soc);
            tagMax  = sprintf('SOC%d_Max',  soc);

            % valOrNaN은 V 단위 → 1e3 곱해서 mV 로 변환
            RMSE_matrix(rowBase+1, ci) = valOrNaN(Tcell, tagMean, 'RMSE') * 1e3; % avg
            RMSE_matrix(rowBase+2, ci) = valOrNaN(Tcell, tagMin,  'RMSE') * 1e3; % min
            RMSE_matrix(rowBase+3, ci) = valOrNaN(Tcell, tagMax,  'RMSE') * 1e3; % max
        end
    end

    Tbl_RMSE = array2table(RMSE_matrix, ...
        'RowNames',      cellstr(RMSE_rowNames), ...
        'VariableNames', rowNames_cells);

    Tbl_RMSE.Properties.Description = ...
        '행: SOC별 RMSE (avg/min/max, mV) / 열: 셀(파일명)';

    %% === (NEW) 선택 SOC에서 avg/min/max × 셀 요약 엑셀 생성 =========
    %   - 네가 보내준 표 형태에 맞게 숫자만 바로 복붙 가능
    %   - 행(15개): R0/R1/R2/τ1/τ2 × (avg, min, max)
    %   - 열: #1, #2, ...  (cellRawNames 순서대로)

    socListStr = {'SOC90','SOC70','SOC50','SOC30'};

    try
        [idxSel, tf] = listdlg( ...
            'PromptString','어떤 SOC 파라미터를 정리할까요?', ...
            'SelectionMode','single', ...
            'ListString',socListStr, ...
            'InitialValue',2);   % 기본 SOC70

        if ~tf
            warning('선택이 취소되어 SOC70을 사용합니다.');
            idxSel = 2;
        end
    catch
        % GUI 안 되는 환경 대비 콘솔 입력
        fprintf('정리할 SOC 선택: 1=SOC90, 2=SOC70, 3=SOC50, 4=SOC30 [기본=2] : ');
        tmp = input('','s');
        v = str2double(tmp);
        if isnan(v) || v<1 || v>4
            v = 2;
        end
        idxSel = v;
    end

    socSelVal = SOC_list(idxSel);
    fprintf('>> 선택된 SOC: %s (=%d%%)\n', socListStr{idxSel}, socSelVal);

    % 행/열 레이블
    statTags_full   = {'Mean','Min','Max'};
    statTags_short  = {'avg','min','max'};
    pNames_noRMSE   = {'R0','R1','R2','tau1','tau2'};  % 5개
    nP2             = numel(pNames_noRMSE);
    nStat2          = numel(statTags_full);
    nRows_total     = nP2 * nStat2;    % 5×3 = 15

    rowLabels = cell(nRows_total,1);
    summaryMat = nan(nRows_total, nCells);

    for pi = 1:nP2
        for si2 = 1:nStat2
            rowIdx = (pi-1)*nStat2 + si2;

            % 행 이름: R0_avg, R0_min, ... , tau2_max
            rowLabels{rowIdx} = sprintf('%s_%s', pNames_noRMSE{pi}, statTags_short{si2});

            rowTag = sprintf('SOC%d_%s', socSelVal, statTags_full{si2});  % 예: SOC70_Mean

            for ci2 = 1:nCells
                key_field = matlab.lang.makeValidName(cellRawNames{ci2});
                Tcell2    = all_summary.(key_field);   % 이 셀의 12×6 요약 table

                summaryMat(rowIdx, ci2) = valOrNaN(Tcell2, rowTag, pNames_noRMSE{pi});
            end
        end
    end

    % 열 이름: #1, #2, ...
    colLabels = arrayfun(@(k)sprintf('#%d',k), 1:nCells, 'UniformOutput', false);

    Tbl_SOC = array2table(summaryMat, ...
        'RowNames',      rowLabels, ...
        'VariableNames', colLabels);

    xlsx_soc = fullfile(save_path, sprintf('Param_avgminmax_byCell_SOC%d.xlsx', socSelVal));
    if exist(xlsx_soc,'file'), delete(xlsx_soc); end

    % 엑셀로 저장 (RowNames 포함)
    writetable(Tbl_SOC, xlsx_soc, 'WriteRowNames', true);

    fprintf('→ 선택 SOC=%d 에 대한 avg/min/max 요약 엑셀 생성 완료: %s\n', ...
        socSelVal, xlsx_soc);

end

results_file = fullfile(save_path,'2RC_results.mat');
if exist('Tbl_ECM_mean','var')
    save(results_file, 'all_para_hats','all_summary','Tbl_ECM_mean','Tbl_RMSE','-v7.3');
else
    save(results_file, 'all_para_hats','all_summary','-v7.3');
end
fprintf('2RC 결과 저장 완료: %s\n', results_file);

% ── 보조 함수 ─────────────────────────────────────────────────────────
function cost = RMSE_2RC(V_true, para, t, I, OCV)
    V_est = RC_model_2(para,t,I,OCV);    
    cost  = sqrt(mean((V_true - V_est).^2));
end

function y = valOrNaN(T, rowName, colName)
    if ismember(rowName, T.Properties.RowNames)
        y = T{rowName, colName};
        if isempty(y), y = NaN; end
    else
        y = NaN;
    end
end
