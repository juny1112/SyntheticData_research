%% ================================================================
%  Compare 5 runs: tau1/tau2 median-based outlier detection
%   - tau1 threshold: 5s
%   - tau2 threshold: 10s
%   - report: cell + SOC + load
% ================================================================
clc; clear;

%% ---- user settings ----
folder_SIM = 'G:\공유 드라이브\BSL_Data4\HNE_RPT_@50,70_251214_9\Driving\SIM_parsed\20degC\이름정렬';

fit_window_sec = 600;

nRuns = 10;                 % 총 n번 fitting 결과 비교
tau1_thr = 5;              % [s]
tau2_thr = 10;             % [s]

% 결과 폴더 규칙: 1_TS_2RC_fitting_%ds, 2_TS_2RC_fitting_%ds, ...
save_root = fileparts(folder_SIM);  % -> ...\SIM_parsed

% 부하 이름(너 코드 가정과 동일)
loadNames = {'US06','UDDS','HWFET','WLTP','CITY1','CITY2','HW1','HW2'};
nLoads    = numel(loadNames);

% (중요) SOC_list는 각 run 결과 파일에서 읽어오도록 했음
% (없으면 아래에서 fallback으로 NaN 처리)

%% ---- load all runs ----
Runs = struct([]);
for r = 1:nRuns
    run_dir = fullfile(save_root, sprintf('%d_TS_2RC_fitting_%ds', r, fit_window_sec));

    % 너 코드의 저장 파일명 규칙
    results_file = fullfile(run_dir, sprintf('2RC_results_%ds.mat', fit_window_sec));

    if ~exist(results_file, 'file')
        error("결과 파일 없음: %s", results_file);
    end

    S = load(results_file); % all_para_hats, all_load_idx 등이 있어야 함

    if ~isfield(S,'all_para_hats') || ~isfield(S,'all_load_idx')
        error("파일에 all_para_hats / all_load_idx 없음: %s", results_file);
    end

    Runs(r).dir = run_dir;
    Runs(r).file = results_file;
    Runs(r).all_para_hats = S.all_para_hats;
    Runs(r).all_load_idx  = S.all_load_idx;

    % SOC_list가 저장되어 있으면 사용(없으면 빈칸)
    if isfield(S,'SOC_list')
        Runs(r).SOC_list = S.SOC_list;
    else
        Runs(r).SOC_list = [];
    end
end

%% ---- determine cell list as intersection across runs (robust) ----
cells_r1 = fieldnames(Runs(1).all_para_hats);
cells_common = cells_r1;

for r = 2:nRuns
    cells_common = intersect(cells_common, fieldnames(Runs(r).all_para_hats));
end

if isempty(cells_common)
    error("공통 셀(field)이 없습니다. 각 run의 all_para_hats fieldnames 확인 필요.");
end

%% ---- SOC_list resolution (assume same across runs) ----
SOC_list = Runs(1).SOC_list;
if isempty(SOC_list)
    warning("SOC_list가 결과 파일에 저장되어 있지 않습니다. SOC 라벨은 'SOC#idx'로 표기합니다.");
    % SOC 개수는 all_load_idx에서 유추
    tmp = Runs(1).all_load_idx.(cells_common{1});
    nSOC = size(tmp,1);
    SOC_list = (1:nSOC); % fallback (의미없는 번호)
    soc_is_fallback = true;
else
    tmp = Runs(1).all_load_idx.(cells_common{1});
    nSOC = size(tmp,1);
    soc_is_fallback = false;
end

%% ---- build long table of flagged cases ----
FlagTable = table();

rowk = 0;

for ci = 1:numel(cells_common)
    cell_field = cells_common{ci};                 % valid fieldname
    base_raw   = cell_field;                       % 네가 base_raw를 makeValidName 했으니 일단 이렇게 둠
    % (원하면 여기서 base_raw 복원 규칙 적용 가능)

    % SOC×Load×Run마다 tau 수집용
    for si = 1:nSOC
        for li = 1:nLoads

            tau1_runs = nan(nRuns,1);
            tau2_runs = nan(nRuns,1);

            % run별 파라미터 뽑기
            for r = 1:nRuns
                if ~isfield(Runs(r).all_load_idx, cell_field)
                    continue
                end
                load_idx_mat = Runs(r).all_load_idx.(cell_field); % [nSOC×nLoads]
                if si > size(load_idx_mat,1) || li > size(load_idx_mat,2)
                    continue
                end
                sIdx = load_idx_mat(si, li); % 해당 SOC/부하의 seg index

                if isnan(sIdx) || sIdx < 1
                    continue
                end

                P = Runs(r).all_para_hats.(cell_field); % [nSeg×8], 1:5=[R0 R1 R2 tau1 tau2]
                if sIdx > size(P,1)
                    continue
                end

                tau1_runs(r) = P(sIdx,4);
                tau2_runs(r) = P(sIdx,5);
            end

            % 유효 run 개수
            valid = isfinite(tau1_runs) & isfinite(tau2_runs);
            if nnz(valid) < 3
                % 최소 3개 이상은 있어야 median 비교가 의미있어서 skip (원하면 2로 낮춰도 됨)
                continue
            end

            med_tau1 = median(tau1_runs(valid));
            med_tau2 = median(tau2_runs(valid));

            % run별 deviation 체크 후, threshold 넘는 run만 기록
            for r = 1:nRuns
                if ~valid(r), continue; end

                d1 = abs(tau1_runs(r) - med_tau1);
                d2 = abs(tau2_runs(r) - med_tau2);

                if (d1 >= tau1_thr) || (d2 >= tau2_thr)
                    rowk = rowk + 1;

                    if soc_is_fallback
                        soc_txt = sprintf('SOC_idx%d', si);
                    else
                        soc_txt = sprintf('SOC%d', SOC_list(si));
                    end

                    FlagTable.row_id(rowk,1)   = rowk;
                    FlagTable.cell(rowk,1)     = string(base_raw);
                    FlagTable.SOC(rowk,1)      = string(soc_txt);
                    FlagTable.load(rowk,1)     = string(loadNames{li});
                    FlagTable.run(rowk,1)      = r;

                    FlagTable.tau1(rowk,1)     = tau1_runs(r);
                    FlagTable.tau2(rowk,1)     = tau2_runs(r);
                    FlagTable.med_tau1(rowk,1) = med_tau1;
                    FlagTable.med_tau2(rowk,1) = med_tau2;

                    FlagTable.d_tau1(rowk,1)   = d1;
                    FlagTable.d_tau2(rowk,1)   = d2;

                    % 어떤 조건으로 걸렸는지
                    FlagTable.flag_tau1(rowk,1)= (d1 >= tau1_thr);
                    FlagTable.flag_tau2(rowk,1)= (d2 >= tau2_thr);
                end
            end

        end
    end
end

%% ---- show & save ----
if isempty(FlagTable)
    fprintf("✅ 이상치 없음: (|tau1-med|<%g, |tau2-med|<%g) for all checked items\n", tau1_thr, tau2_thr);
else
    % 보기 좋게 정렬
    FlagTable = sortrows(FlagTable, {'cell','SOC','load','run'});

    disp("🚩 Median 대비 tau 이상치 감지 결과:");
    disp(FlagTable);

    out_csv = fullfile(save_root, sprintf('TauOutliers_median_%druns_%ds.csv', nRuns, fit_window_sec));
    try
        writetable(FlagTable, out_csv);
        fprintf("Saved: %s\n", out_csv);
    catch ME
        warning("CSV 저장 실패: %s", ME.message);
    end
end
