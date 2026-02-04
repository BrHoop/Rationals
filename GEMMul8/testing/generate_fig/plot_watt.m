function plot_watt(saveflag,savedir)

arguments (Input)
    saveflag (1,1) logical = false
    savedir (1,1) string = ""
end

%% files
fileList_f = dir("oz2_results_f_watt_NVIDIA_*");
filename_f = strings(length(fileList_f),1);
for i=1:length(fileList_f)
    filename_f(i) = string(fileList_f(i).name);
end
fileList_d = dir("oz2_results_d_watt_NVIDIA_*");
filename_d = strings(length(fileList_d),1);
for i=1:length(fileList_d)
    filename_d(i) = string(fileList_d(i).name);
end
FontSize = 8;

%% float
fig = figure;
fig.Position(1) = 100;
fig.Position(3) = 1000;
fig.Position(4) = 125*length(filename_f);
t = tiledlayout(length(filename_f),6);
for fn = 1:length(filename_f)
    filename = filename_f(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 2;
    n = readmatrix(filename,opts);
    n_list = unique(n,'stable');
    opts.SelectedVariableNames = 5;
    func = string(readmatrix(filename,opts));
    func_list = unique(func,'stable');
    opts.SelectedVariableNames = 8;
    watt = readmatrix(filename,opts);
    opts.SelectedVariableNames = 9;
    gflops_watt = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 15;
    xx = moduli_min:moduli_max;
    XLIM = [2 12];

    for ni = 1:length(n_list)
        nexttile(ni + 6*(fn-1)); hold on;
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.XAxis.FontSize = FontSize-2;
        ax.XAxis.TickDirection = 'out';
        ax.YAxis.TickDirection = 'out';
        xtickangle(0);
        SGEMM    = gflops_watt(n == n_list(ni) & func == "SGEMM");
        SGEMM_TF = gflops_watt(n == n_list(ni) & func == "SGEMM-TF32");
        SGEMM_BF = gflops_watt(n == n_list(ni) & func == "SGEMM-BF16X9");
        FP16TCEC = gflops_watt(n == n_list(ni) & func == "FP16TCEC_SCALING");
        OS2_fast = gflops_watt(n == n_list(ni) & contains(func,"OS2-fast"));
        OS2_accu = gflops_watt(n == n_list(ni) & contains(func,"OS2-accu"));
        plot(xx, SGEMM*ones(size(xx)), mark(1,1), 'DisplayName', "SGEMM", 'LineWidth',1);
        plot(xx, SGEMM_TF*ones(size(xx)), mark(1,2), 'DisplayName', "TF32GEMM", 'LineWidth',1);
        if ~isempty(SGEMM_BF)
            plot(xx, SGEMM_BF*ones(size(xx)), mark(1,5), 'DisplayName', "BF16x9", 'LineWidth',1);
        end
        if ~isempty(FP16TCEC)
            plot(xx, FP16TCEC*ones(size(xx)), mark(1,6), 'DisplayName', "cuMpSGEMM", 'LineWidth',1);
        end
        plot(xx, OS2_fast, mark(1,3), 'DisplayName', "OS II-fast", 'LineWidth',1);
        plot(xx, OS2_accu, mark(1,4), 'DisplayName', "OS II-accu", 'LineWidth',1);

        xr = xregion(6,9,FaceColor="r");
        set(get(get(xr, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');

        set(gca,'FontSize',FontSize);
        xlim(XLIM);
        xticks(XLIM(1):XLIM(2));
        str = string(XLIM(1):XLIM(2));
        str(1:2:end) = "";
        xticklabels(str)
        ylim('padded');
        yl = ylim;
        for inc = [200,150,100,50,25,10,5,2,1]
            yticks(0:inc:yl(2));
            yt = yticks;
            if length(yt) >= 4
                break
            end
        end
        title("n=" + n_list(ni),'FontSize',FontSize+2);
    end
    lgd = legend;
    lgd.Layout.Tile = 6 * fn;
    lgd.NumColumns = 1;
    lgd.FontSize = FontSize+2;
    pattern = "watt_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    if contains(env,"GH200")
        title(lgd, "GH200", 'FontSize',FontSize+2);
    elseif contains(env,"A100")
        title(lgd, "A100 SXM4", 'FontSize',FontSize+2);
    elseif contains(env,"RTX 4090")
        title(lgd, "RTX 4090", 'FontSize',FontSize+2);
    elseif contains(env,"RTX 5080")
        title(lgd, "RTX 5080", 'FontSize',FontSize+2);
    else
        title(t, env, 'FontSize',FontSize);
    end
end
xlabel(t,"Number of moduli",'FontSize',FontSize+2);
ylabel(t,"GFLOPS/watt",'FontSize',FontSize+2);
t.TileSpacing = "tight";
t.Padding = "compact";

if saveflag
    pattern = "(.*?)_NVIDIA";
    match = regexp(filename, pattern, 'tokens');
    figname = match{1}{1};
    savefig(fig,savedir+figname);
    exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
end

%% double
fig = figure;
fig.Position(1) = 100;
fig.Position(3) = 1000;
fig.Position(4) = 125*length(filename_d);
t = tiledlayout(length(filename_d),6);
for fn = 1:length(filename_d)
    filename = filename_d(fn);
    opts = detectImportOptions(filename);
    opts.SelectedVariableNames = 2;
    n = readmatrix(filename,opts);
    n_list = unique(n,'stable');
    opts.SelectedVariableNames = 5;
    func = string(readmatrix(filename,opts));
    func_list = unique(func,'stable');
    opts.SelectedVariableNames = 8;
    watt = readmatrix(filename,opts);
    opts.SelectedVariableNames = 9;
    gflops_watt = readmatrix(filename,opts);
    moduli_min = 2;
    moduli_max = 20;
    xx = moduli_min:moduli_max;
    XLIM = [8 moduli_max];

    for ni = 1:length(n_list)
        nexttile(ni + 6*(fn-1)); hold on;
        set( groot, 'defaultAxesTickDir', 'out' )
        ax = gca;
        ax.XGrid = 'on';
        ax.YGrid = 'on';
        ax.XAxis.FontSize = FontSize-2;
        ax.XAxis.TickDirection = 'out';
        ax.YAxis.TickDirection = 'out';
        xtickangle(0);
        DGEMM = gflops_watt(n == n_list(ni) & func == "DGEMM");
        ozIMMU_EF_8 = gflops_watt(n == n_list(ni) & contains(func,"ozIMMU_EF-8"));
        ozIMMU_EF_9 = gflops_watt(n == n_list(ni) & contains(func,"ozIMMU_EF-9"));
        OS2_fast = gflops_watt(n == n_list(ni) & contains(func,"OS2-fast"));
        OS2_accu = gflops_watt(n == n_list(ni) & contains(func,"OS2-accu"));
        plot(xx, DGEMM*ones(size(xx)), mark(1,1), 'DisplayName', "DGEMM", 'LineWidth',1);
        if ~isempty(ozIMMU_EF_8)
            plot(xx, ozIMMU_EF_8*ones(size(xx)), mark(1,2), 'DisplayName', "ozIMMU\_EF-8", 'LineWidth',1);
        end
        if ~isempty(ozIMMU_EF_9)
            plot(xx, ozIMMU_EF_9*ones(size(xx)), mark(1,5), 'DisplayName', "ozIMMU\_EF-9", 'LineWidth',1);
        end
        plot(xx, OS2_fast, mark(1,3), 'DisplayName', "OS II-fast", 'LineWidth',1);
        plot(xx, OS2_accu, mark(1,4), 'DisplayName', "OS II-accu", 'LineWidth',1);

        xr = xregion(14,18,FaceColor="r");
        set(get(get(xr, 'Annotation'), 'LegendInformation'), 'IconDisplayStyle', 'off');

        xlim(XLIM);
        xticks(XLIM(1):XLIM(2));
        str = string(XLIM(1):XLIM(2));
        str(1:2:end) = "";
        xticklabels(str)
        ylim('padded')
        yl = ylim;
        for inc = [200,150,100,50,25,10,5,2,1]
            yticks(0:inc:yl(2));
            yt = yticks;
            if length(yt) >= 4
                break
            end
        end
        set(gca,'FontSize',FontSize);
        title("n=" + n_list(ni),'FontSize',FontSize+2);
    end
    lgd = legend;
    lgd.Layout.Tile = 6*fn;
    lgd.NumColumns = 1;
    lgd.FontSize = FontSize+2;
    pattern = "watt_(.*?)_2025";
    match = regexp(filename, pattern, 'tokens');
    env = match{1}{1};
    env = replace(env,"_"," ");
    env = replace(env,"-"," ");
    if contains(env,"GH200")
        title(lgd, "GH200", 'FontSize',FontSize+2);
    elseif contains(env,"A100")
        title(lgd, "A100 SXM4", 'FontSize',FontSize+2);
    elseif contains(env,"RTX 4090")
        title(lgd, "RTX 4090", 'FontSize',FontSize+2);
    elseif contains(env,"RTX 5080")
        title(lgd, "RTX 5080", 'FontSize',FontSize+2);
    else
        title(t, env, 'FontSize',FontSize+2);
    end
end
xlabel(t,"Number of moduli",'FontSize',FontSize+2);
ylabel(t,"GFLOPS/watt",'FontSize',FontSize+2);
t.TileSpacing = "tight";
t.Padding = "compact";

if saveflag
    pattern = "(.*?)_NVIDIA";
    match = regexp(filename, pattern, 'tokens');
    figname = match{1}{1};
    savefig(fig,savedir+figname);
    exportgraphics(fig,savedir+figname + ".png",'Resolution',600);
end

end

